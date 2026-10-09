"""Matched, root-balanced float/QAT numerical pipeline and recovery state."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for variable in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from tools import compact_value_bfm_train as core
from tools import compact_representation_pilot as ranking
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_export_v2 as exporter
from tools import rank_two_focused_checkpoint_v1 as storage

SCHEMA = "papersoccer.focused-training.v1"


@dataclass(frozen=True)
class Architecture:
    profile: str
    hidden_one: int
    hidden_two: int
    inputs: int = 6301

    @property
    def shapes(self):
        return dict(w1=(self.inputs,self.hidden_one),
                    w2=(self.hidden_one,self.hidden_two),w3=(self.hidden_two,))


def initialize(profile, seed):
    if profile not in campaign.FAMILIES:
        raise ValueError("only the two matched families are trainable")
    shape = campaign.FAMILIES[profile]
    architecture = Architecture(profile,shape[1],shape[2])
    return architecture,core.initialize_parameters(architecture,seed)


def quantize(parameters, scales):
    if set(scales) != set(parameters) or any(not math.isfinite(float(s)) or s <= 0 for s in scales.values()):
        raise ValueError("positive finite scales for every layer required")
    integer = {name:np.clip(np.rint(value/np.float32(scales[name])),-7,7).astype(np.int8)
               for name,value in parameters.items()}
    return core.QuantizedWeights(integer,{name:np.float32(value) for name,value in scales.items()})


def forward(parameters, architecture, active, quantized=None):
    """Vectorize across rows while retaining literal float32 order within each."""
    if not active:
        raise ValueError("empty forward batch")
    active = [np.asarray(row,dtype=np.intp) for row in active]
    if any(not len(row) or np.any(row<0) or np.any(row>=architecture.inputs) for row in active):
        raise ValueError("feature index outside model")
    count,max_count=len(active),max(map(len,active))
    ids=np.zeros((count,max_count),dtype=np.intp)
    mask=np.zeros((count,max_count),dtype=np.bool_)
    for index,row in enumerate(active):ids[index,:len(row)]=row;mask[index,:len(row)]=True
    dtype=np.float32 if quantized is None else np.int32
    first_pre=np.zeros((count,architecture.hidden_one),dtype=dtype)
    source=parameters["w1"] if quantized is None else quantized.integer["w1"]
    for index in range(max_count):
        first_pre += np.where(mask[:,index,None],source[ids[:,index]],0).astype(dtype)
    if quantized is not None:
        first_pre=first_pre.astype(np.float32)*quantized.scales["w1"]
    first=core.first_activation(first_pre)
    effective=parameters if quantized is None else quantized.effective()
    second_pre=np.zeros((count,architecture.hidden_two),dtype=np.float32)
    for i in range(architecture.hidden_one):
        for j in range(architecture.hidden_two):
            if quantized is None:term=np.asarray(first[:,i]*parameters["w2"][i,j],dtype=np.float32)
            else:term=np.asarray(np.asarray(first[:,i]*quantized.scales["w2"],dtype=np.float32)*quantized.integer["w2"][i,j],dtype=np.float32)
            second_pre[:,j]=np.asarray(second_pre[:,j]+term,dtype=np.float32)
    second=core.second_activation(second_pre)
    output_pre=np.zeros(count,dtype=np.float32)
    for j in range(architecture.hidden_two):
        if quantized is None:term=np.asarray(second[:,j]*parameters["w3"][j],dtype=np.float32)
        else:term=np.asarray(np.asarray(second[:,j]*quantized.scales["w3"],dtype=np.float32)*quantized.integer["w3"][j],dtype=np.float32)
        output_pre=np.asarray(output_pre+term,dtype=np.float32)
    output=np.asarray([core._fast_tanh_scalar(np.float32(value)) for value in output_pre],dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("nonfinite network output")
    return output,(first_pre,first,second_pre,second,output_pre)


def root_rows(groups):
    active,targets,signs,spans=[],[],[],[]
    for group in groups:
        if type(group["mover"]) is not int or group["mover"] not in (0,1) or type(group["exhaustive"]) is not bool:
            raise ValueError("mover or completeness contract differs")
        start=len(active)
        for row in group["successors"]:
            indices=row["active"]
            if indices!=sorted(set(indices)) or not indices or any(type(index) is not int for index in indices) or not 0<=indices[0]<=indices[-1]<6301:
                raise ValueError("canonical sparse features differ")
            value=row["teacher_value"]
            if type(value) not in (int,float) or not math.isfinite(value) or not -1<=value<=1 or type(row["value_mover"]) is not int or row["value_mover"] not in (0,1):
                raise ValueError("teacher value/perspective differs")
            proof=row.get("proof",{})
            if proof.get("solved"):
                winner=proof.get("proven_winner")
                if type(winner) is not int or winner not in (0,1) or value!=(1. if winner==row["value_mover"] else -1.):
                    raise ValueError("exact proof value differs")
            if row.get("terminal") and abs(value)!=1:
                raise ValueError("terminal value must remain exact")
            active.append(np.asarray(indices,dtype=np.intp));targets.append(value)
            signs.append(1. if row["value_mover"]==group["mover"] else -1.)
        if len(active)==start:
            raise ValueError("empty successor group")
        spans.append(slice(start,len(active)))
    return active,np.asarray(targets,dtype=np.float32),np.asarray(signs,dtype=np.float32),spans


def objective(groups,predictions,targets,signs,spans,ranking_weight=.25):
    if len(groups)!=len(spans) or not groups:
        raise ValueError("group normalization differs")
    gradient=np.zeros_like(predictions);huber=rank_loss=0.
    for group,span in zip(groups,spans,strict=True):
        scalar,local=core._weighted_huber_loss_gradient(predictions[span],targets[span],np.ones(span.stop-span.start,dtype=np.float32))
        # Incomplete enumeration is never promoted to an exhaustive ranking label.
        rank=0.
        if group["exhaustive"] and span.stop-span.start>1:
            rank,derivative=ranking.ranking_loss_gradient(predictions[span],targets[span],signs[span])
            local+=np.float32(ranking_weight)*derivative
        gradient[span]=local/np.float32(len(groups));huber+=scalar/len(groups);rank_loss+=rank/len(groups)
    return huber,rank_loss,gradient


def update_root(parameters,architecture,optimizer,groups,scales=None,ranking_weight=.25):
    active,targets,signs,spans=root_rows(groups)
    quantized=None if scales is None else quantize(parameters,scales)
    predictions,cache=forward(parameters,architecture,active,quantized)
    huber,rank_loss,derivative=objective(groups,predictions,targets,signs,spans,ranking_weight)
    gradients=core._network_gradients(parameters,architecture,active,cache,derivative,
                                     parameters if quantized is None else quantized.effective())
    if any(not np.all(np.isfinite(g)) for g in gradients.values()):
        raise FloatingPointError("nonfinite gradient")
    norm=math.sqrt(sum(float(np.sum(g.astype(np.float64)**2)) for g in gradients.values()))
    if norm>5:
        for gradient in gradients.values():gradient*=np.float32(5/norm)
    optimizer.update(parameters,gradients)
    if any(not np.all(np.isfinite(a)) for a in (*parameters.values(),*optimizer.first.values(),*optimizer.second.values())):
        raise FloatingPointError("nonfinite optimizer state")
    return dict(huber=huber,ranking=rank_loss,objective=huber+.25*rank_loss)


def diagnostics(parameters,cache,scales=None):
    first_pre,first,second_pre,second,output_pre=cache
    result=dict(output_saturation=float(np.mean(np.abs(output_pre)>4.95)),
                first_positive_fraction=float(np.mean(first_pre>=0)),
                second_positive_fraction=float(np.mean(second_pre>=0)))
    if scales is not None:
        q=quantize(parameters,scales);count=sum(value.size for value in parameters.values())
        result.update(zero_code_fraction=sum(int(np.count_nonzero(value==0)) for value in q.integer.values())/count,
                      clipped_weight_fraction=sum(int(np.count_nonzero(np.abs(value)>7*scales[name])) for name,value in parameters.items())/count)
    return result


def calibrate(parameters,architecture,training_roots):
    if not training_roots or any(group.get("split")!="train" for groups in training_roots for group in groups):
        raise ValueError("scale calibration accepts TRAIN only")
    prepared=[]
    for groups in training_roots:
        active,*_=root_rows(groups);prediction,_=forward(parameters,architecture,active)
        prepared.append((active,prediction))
    candidates={name:[np.float32(max(float(np.percentile(np.abs(value),p))/7,1e-8)) for p in (95,99,100)]
                for name,value in parameters.items()}
    scales={name:values[-1] for name,values in candidates.items()};trials=[]
    for pass_index in range(2):
        for layer in ("w1","w2","w3"):
            options=[]
            for candidate in candidates[layer]:
                proposed={**scales,layer:candidate};quantized=quantize(parameters,proposed)
                mse=sum(float(np.mean((forward(parameters,architecture,active,quantized)[0]-prediction)**2))
                        for active,prediction in prepared)/len(prepared)
                options.append((mse,float(candidate),proposed))
                trials.append(dict(pass_index=pass_index,layer=layer,scale=float(candidate),root_balanced_float_mse=mse))
            scales=min(options,key=lambda row:(row[0],row[1]))[2]
    return scales,dict(split="train",passes=2,percentiles=[95,99,100],trials=trials,
                       selected_scales={key:float(value) for key,value in scales.items()})


def stopping(history,minimum,maximum,patience=5,tolerance=1e-4):
    if not history or any(not math.isfinite(value) for value in history):
        raise ValueError("finite objective history required")
    best=float("inf");stale=0
    for value in history:
        if value<best-tolerance:best=value;stale=0
        else:stale+=1
    if len(history)>=maximum:return "budget-limited"
    if len(history)>=minimum and stale>=patience:return "plateau"
    return None


def checkpoint(path,parameters,optimizer,cursor,binding,parent=None):
    return storage.checkpoint(path,parameters,optimizer,cursor,binding,parent)


def restore(receipt,parameters,optimizer,binding):
    return storage.restore(receipt,parameters,optimizer,binding)


def runtime(profile,parameters,scales,lineage):
    quantized=quantize(parameters,scales)
    values=np.concatenate([quantized.integer[name].ravel() for name in ("w1","w2","w3")]).astype(int).tolist()
    return exporter.document(profile,values,[float(scales[name]) for name in ("w1","w2","w3")],lineage)


def validation(corpus,parameters,architecture,scales=None,native=None):
    values=[];comparable=early=0
    for root in sorted(corpus.roots["validation"]):
        groups=corpus.root("validation",root);active,target,signs,spans=root_rows(groups)
        q=None if scales is None else quantize(parameters,scales)
        predictions,cache=forward(parameters,architecture,active,q)
        if native is not None:
            actual=np.asarray(native(root,active),dtype=np.float32)
            if actual.shape!=predictions.shape or not np.all(np.isfinite(actual)) or not np.allclose(actual,predictions,rtol=0,atol=1e-6):
                raise ValueError("source-bound native validation prediction differs")
            predictions=actual
        huber,ranking_loss,_=objective(groups,predictions,target,signs,spans)
        regrets=[];early_regrets=[];flips=[]
        float_values,_=forward(parameters,architecture,active)
        for group,span in zip(groups,spans,strict=True):
            teacher=target[span]*signs[span]
            if not group["exhaustive"] or len(teacher)<2 or float(np.max(teacher)-np.min(teacher))<=1e-4:
                continue
            picked=int(np.argmax(predictions[span]*signs[span]));float_picked=int(np.argmax(float_values[span]*signs[span]))
            regret=float(np.max(teacher)-teacher[picked]);regrets.append(regret);flips.append(picked!=float_picked)
            if group["edges"]<=12:early_regrets.append(regret)
        comparable+=bool(regrets);early+=bool(early_regrets)
        values.append(dict(huber=huber,ranking=ranking_loss,objective=huber+.25*ranking_loss,
                           regret=None if not regrets else float(np.mean(regrets)),
                           early_regret=None if not early_regrets else float(np.mean(early_regrets)),
                           quantized_action_flip_fraction=None if not flips else float(np.mean(flips)),
                           diagnostics=diagnostics(parameters,cache,scales)))
    if comparable<100 or early<100:raise ValueError("insufficient comparable independent validation roots")
    return {**{name:float(np.mean([row[name] for row in values if row[name] is not None]))
               for name in ("huber","ranking","objective","regret","early_regret","quantized_action_flip_fraction")},
            "validation_roots":len(values),"comparable_roots":comparable,"early_roots":early,
            "diagnostics":{name:float(np.mean([row["diagnostics"][name] for row in values])) for name in values[0]["diagnostics"]}}


def validate_corpus_binding(plan, corpus):
    """Bind the merged512TRAIN/freshVALIDATION corpus and all predecessor work."""
    from tools import rank_two_focused_expansion_label_data_v1 as pipeline
    from tools import rank_two_focused_labels_v4 as expected_labels
    from tools import rank_two_focused_expansion_corpus_v1 as expected_merger
    data = pipeline.data
    pipeline.check_current(plan, "train")
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    manifest = corpus.manifest
    if (manifest.get("generation_plan") != plan["generation_plan"]
            or manifest.get("expansion_declaration") != plan["declaration"]
            or manifest.get("previous_corpus") != plan["previous_corpus"]
            or manifest.get("old_validation_reserved") is not True
            or manifest.get("outcomes_as_targets") is not False):
        raise ValueError("expanded corpus declaration/source binding differs")
    label_path = data.bound(manifest["label_plan"])
    label_plan = campaign.read(label_path)
    pipeline.verify_teacher_plan(label_plan)
    if (plan["label_plan"] != campaign.record(label_path)
            or label_plan["previous_corpus"] != plan["previous_corpus"]
            or plan["teacher"] != declaration["teacher"]
            or label_plan["teacher"] != plan["teacher"]
            or plan["label_producer"] != campaign.record(expected_labels.__file__)):
        raise ValueError("expanded teacher plan/source differs")
    execution_path = campaign.verify(manifest["execution"])
    execution = campaign.read(execution_path)
    execution_recipe = campaign.read(Path(execution_path).parent / "EXECUTION.json")
    if (plan["label_execution"] != campaign.record(execution_path)
            or execution["plan"] != plan["label_plan"] or execution.get("passed") is not True
            or execution_recipe["plan"] != plan["label_plan"]
            or execution_recipe["producer"] != plan["label_producer"]):
        raise ValueError("expanded teacher execution binding differs")
    merge_plan = campaign.read(campaign.verify(manifest["expansion_merge_plan"]))
    archive = campaign.read(campaign.verify(plan["corpus_archive"]))
    if (merge_plan["producer"] != campaign.record(expected_merger.__file__)
            or archive.get("passed") is not True or archive["corpus"] != plan["corpus"]
            or archive["teacher_execution"] != plan["label_execution"]
            or archive["training_validation_reflected_features_disjoint"] is not True):
        raise ValueError("expanded corpus merge/archive binding differs")
    previous = campaign.read(campaign.verify(plan["previous_corpus"]))
    expected_previous = [row for row in previous["group_artifacts"] if row["split"] == "train"]
    retained_ids = set(manifest["retained_train_root_group_ids"])
    actual_previous = [row for row in manifest["group_artifacts"] if row["root_group_id"] in retained_ids]
    if sorted(expected_previous,key=lambda row:row["group_id"]) != sorted(actual_previous,key=lambda row:row["group_id"]):
        raise ValueError("useful previous training labels were changed or discarded")
    generation = campaign.read(campaign.verify(plan["generation_plan"]))
    bank = campaign.read(campaign.verify(generation["bank"]))
    fresh_train = {row["root_group_id"] for row in bank["rows"] if row["split"] == "train"}
    fresh_validation = {row["root_group_id"] for row in bank["rows"] if row["split"] == "validation"}
    reserved = {row["root_group_id"] for row in previous["group_artifacts"] if row["split"] == "validation"}
    if (len(retained_ids) != 128 or len(fresh_train) != 384 or len(fresh_validation) != 128
            or set(corpus.roots["train"]) != retained_ids | fresh_train
            or len(corpus.roots["train"]) != 512
            or not set(corpus.roots["validation"]) <= fresh_validation
            or set(corpus.roots["validation"]) != set(manifest["eligible_validation_root_group_ids"])
            or len(corpus.roots["validation"]) < 100
            or reserved & (set(corpus.roots["train"]) | set(corpus.roots["validation"]))):
        raise ValueError("declared512TRAIN/independent freshVALIDATION membership differs")
    ancestry = campaign.read(campaign.verify(plan["ancestry_ready"]))
    if ancestry.get("passed") is not True or any(ref not in ancestry.get("exposures",[]) for ref in plan["required_exposures"]):
        raise ValueError("new trajectories, teacher branches and recipes lack incremental ancestry")
    return True


def warm_start(plan, binding, parameters, optimizer, out):
    """Restore all parent tensors/moments and begin a full, new-plan anchor."""
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    parent = declaration["preserve_checkpoint"]
    if (plan["parent_plan"] != declaration["parent_plan"]
            or plan["parent_checkpoint"] != parent
            or parent["plan"] != plan["parent_plan"]
            or plan["seed"] != declaration["parent_seed"]
            or plan["parent_epoch_receipt"] != declaration["preserve_parent_epoch_receipt"]):
        raise ValueError("exact useful float parent/seed binding differs")
    epoch = campaign.read(campaign.verify(plan["parent_epoch_receipt"]))
    if epoch["checkpoint"] != parent or epoch["phase"] != "float":
        raise ValueError("warm start is not the preserved float checkpoint")
    campaign.verify(plan["parent_plan"])
    cursor = restore(parent, parameters, optimizer, plan["parent_plan"])
    directory = out / "warm-start"
    path = directory / "checkpoint.psc"
    receipt_path = path.with_suffix(".json")
    if receipt_path.exists():
        anchor = campaign.read(receipt_path)
        restore(anchor, parameters, optimizer, binding)
    else:
        if (directory / "CLAIM.json").exists():
            raise RuntimeError("unknown warm-start anchor claim retained")
        storage.require_headroom(path, sum(value.nbytes for value in storage.arrays(parameters,optimizer).values()))
        campaign.immutable(directory / "CLAIM.json", dict(plan=binding, parent_plan=plan["parent_plan"],
            parent_checkpoint=parent["checkpoint"], parent_cursor=cursor, all_optimizer_moments_preserved=True))
        anchor = checkpoint(path, parameters, optimizer, cursor, binding, parent=None)
    if (anchor["storage"]["mode"] != "full" or anchor["storage"]["parent"] is not None
            or anchor["storage"]["raw_sha256"] != parent["storage"]["raw_sha256"]
            or anchor["optimizer_step"] != parent["optimizer_step"] or anchor["cursor"] != parent["cursor"]):
        raise ValueError("full warm-start anchor does not preserve all parent arrays/step/cursor")
    campaign.immutable(out / "WARM_START.json", dict(plan=binding, parent_plan=plan["parent_plan"],
        parent_epoch_receipt=plan["parent_epoch_receipt"], parent_checkpoint=parent["checkpoint"],
        anchor=anchor, exact_parameters_and_optimizer=True, parent_cursor_retained=True,
        next_float_rng_seed=cursor["rng_seed"] + 1, first_new_plan_checkpoint_full=True,
        predecessor_storage_and_checkpoints_unchanged=True))
    return anchor


def fit(plan_path):
    """Execute one declared data-expansion arm from its exact useful float parent."""
    from tools import compact_representation_train as corpus_reader
    from tools import rank_two_focused_native_v3 as native_probe
    import importlib.util
    plan=campaign.read(plan_path);binding=campaign.record(plan_path)
    if plan.get("schema")!=SCHEMA+".plan" or plan.get("recipe")!=campaign.RECIPE or plan.get("profile")!="dd8" or plan.get("seed")!=2026100702:
        raise ValueError("declared single-family expansion recipe/profile/seed differs")
    if plan.get("producer")!=campaign.record(__file__) or plan.get("numpy_version")!=np.__version__ or plan.get("python")!=campaign.record(Path(sys.executable).resolve()):
        raise ValueError("numerical producer/runtime differs")
    if plan.get("checkpoint_storage")!=campaign.record(storage.__file__):raise ValueError("lossless checkpoint producer differs")
    for reference in plan["inputs"].values():campaign.verify(reference)
    current=Path(plan["campaign_current"]);state=campaign.read(current)
    guard_spec=importlib.util.spec_from_file_location("focused_bound_control",campaign.verify(state["control_guard"]))
    guard=importlib.util.module_from_spec(guard_spec);guard_spec.loader.exec_module(guard)
    from tools import rank_two_focused_work_v1 as work
    work.checked(plan,"train",launch=True)
    if plan["activation"]!=state["focused_activation"]:raise ValueError("campaign activation differs")
    generation=campaign.read(campaign.verify(plan["generation_plan"]))
    declaration=campaign.read(campaign.verify(plan["declaration"]))
    if (generation.get("seed")!=declaration["root_generation_seed"]
            or generation.get("declaration")!=plan["declaration"]
            or declaration["principal_intervention"]!="training-coverage-expansion"
            or declaration["architecture"]!=[6301,8,8,1]):
        raise ValueError("declared expansion generation/family identity differs")
    ancestry=campaign.read(campaign.verify(plan["ancestry_ready"]))
    if ancestry.get("passed") is not True:raise ValueError("fresh training ancestry admission missing")
    corpus=corpus_reader.Corpus(campaign.verify(plan["corpus"]))
    validate_corpus_binding(plan,corpus)
    if len(corpus.roots["train"])!=512 or len(corpus.roots["validation"])<100:
        raise ValueError("expanded512TRAIN/freshVALIDATION root allocation differs")
    architecture=Architecture("dd8",8,8)
    parameters={name:np.zeros(shape,dtype=np.float32) for name,shape in architecture.shapes.items()}
    optimizer=core.AdamW(parameters,learning_rate=.001,weight_decay=1e-5)
    out=Path(plan["output"]);out.mkdir(parents=True,exist_ok=True)
    forbidden=exporter.macro_names();root_ids=sorted(corpus.roots["train"])
    histories={"float":[],"qat":[]};receipts={"float":[],"qat":[]};scales=None
    checkpoint_parent=warm_start(plan,binding,parameters,optimizer,out)
    def pause_check():
        work.checked(plan,"train")
    for phase,minimum,maximum,rate in (("float",10,40,.001),("qat",8,16,.00025)):
        if phase=="qat":
            chosen=min(receipts["float"],key=lambda r:(r["validation"]["objective"],r["epoch"]))
            restore(chosen["checkpoint"],parameters,optimizer,binding)
            checkpoint_parent=chosen["checkpoint"]
            scales,calibration=calibrate(parameters,architecture,[corpus.root("train",root) for root in root_ids])
            campaign.immutable(out/"CALIBRATION.json",{**calibration,"float_checkpoint":chosen["checkpoint"]["checkpoint"],"plan":binding})
        optimizer.learning_rate=np.float32(rate)
        for epoch in range(maximum):
            pause_check();directory=out/phase/f"epoch-{epoch:02d}";directory.mkdir(parents=True,exist_ok=True)
            if (directory/"receipt.json").exists():
                receipt=campaign.read(directory/"receipt.json");restore(receipt["checkpoint"],parameters,optimizer,binding)
                checkpoint_parent=receipt["checkpoint"]
            else:
                rng_seed=(plan["parent_checkpoint"]["cursor"]["rng_seed"]+1+epoch
                          if phase=="float" else plan["seed"]+10000+epoch)
                order=[root_ids[i] for i in np.random.default_rng(rng_seed).permutation(len(root_ids))]
                campaign.immutable(directory/"ORDER.json",dict(plan=binding,phase=phase,epoch=epoch,rng_seed=rng_seed,roots=order))
                final=None
                for index,root in enumerate(order):
                    pause_check();step=directory/f"root-{index:03d}";step.mkdir(exist_ok=True)
                    path=step/"checkpoint.psc";receipt_path=path.with_suffix(".json")
                    if receipt_path.exists():
                        final=campaign.read(receipt_path);restore(final,parameters,optimizer,binding);checkpoint_parent=final;continue
                    if (step/"CLAIM.json").exists():raise RuntimeError("uncommitted root claim retained; explicit recovery required")
                    storage.require_headroom(path,sum(value.nbytes for value in storage.arrays(parameters,optimizer).values()))
                    campaign.immutable(step/"CLAIM.json",dict(plan=binding,phase=phase,epoch=epoch,root=root,index=index))
                    update_root(parameters,architecture,optimizer,corpus.root("train",root),scales)
                    final=checkpoint(path,parameters,optimizer,dict(phase=phase,epoch=epoch,next_root=index+1,rng_seed=rng_seed,root_order=order),binding,parent=checkpoint_parent)
                    checkpoint_parent=final
                float_metrics=validation(corpus,parameters,architecture)
                metrics=float_metrics;native_receipt=None;source_ref=None;runtime_ref=None
                if phase=="qat":
                    doc=runtime(plan["profile"],parameters,scales,dict(kind="focused-production",production_eligible=True,plan=binding,phase=phase,epoch=epoch,seed=plan["seed"]))
                    source,report=exporter.export(doc,forbidden)
                    campaign.immutable(directory/"submission.cpp",source.encode());campaign.immutable(directory/"runtime.json",doc);campaign.immutable(directory/"EXPORT.json",report)
                    binary=native_probe.build(source,report,directory/"native-validation")
                    def native(root,active):
                        requests=["eval "+str(len(row))+" "+" ".join(map(str,row)) for row in active]
                        answers=native_probe.query(binary,requests)
                        if any(answer.startswith("ERROR") for answer in answers):raise ValueError("native validation query failed")
                        return [float(answer) for answer in answers]
                    metrics=validation(corpus,parameters,architecture,scales,native)
                    native_receipt=dict(passed=True,binary=campaign.record(binary),source=campaign.record(directory/"submission.cpp"),roots=metrics["validation_roots"],queries_only=True,source_below_limit=report["below_hard_limit"])
                    campaign.immutable(directory/"NATIVE_PARITY.json",native_receipt)
                    source_ref=campaign.record(directory/"submission.cpp");runtime_ref=campaign.record(directory/"runtime.json")
                receipt=dict(schema=SCHEMA+".epoch",plan=binding,phase=phase,epoch=epoch,checkpoint=final,
                             validation=metrics,float_validation=float_metrics,native=native_receipt,source=source_ref,runtime=runtime_ref)
                campaign.immutable(directory/"receipt.json",receipt)
            if receipt["plan"]!=binding or receipt["phase"]!=phase or receipt["epoch"]!=epoch:
                raise ValueError("completed epoch binding differs")
            receipts[phase].append(receipt);histories[phase].append(receipt["validation"]["objective"])
            reason=stopping(histories[phase],minimum,maximum)
            print(json.dumps(dict(phase=phase,epoch=epoch,metrics=receipt["validation"],stop_reason=reason)),flush=True)
            if reason:
                campaign.immutable(out/(phase.upper()+"_STOP.json"),dict(reason=reason,epochs=len(histories[phase]),converged=False,plateau=reason=="plateau",plan=binding))
                break
    eligible=[receipt for receipt in receipts["qat"] if receipt["native"]["source_below_limit"]]
    selected=min(eligible,key=lambda r:(r["validation"]["regret"],r["validation"]["huber"],r["epoch"])) if eligible else None
    result=dict(schema=SCHEMA+".result",plan=binding,status="complete",selected=selected,
                float_epochs=len(receipts["float"]),qat_epochs=len(receipts["qat"]),optimizer_steps=optimizer.step,
                new_games=0,playing_strength_qualified=False,
                parent_plan=plan["parent_plan"],parent_checkpoint=plan["parent_checkpoint"]["checkpoint"],
                training_roots=512,validation_roots=len(corpus.roots["validation"]))
    campaign.immutable(out/"result.json",result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args();result=fit(args.plan)
    print(json.dumps({key:result[key] for key in ("status","float_epochs","qat_epochs","optimizer_steps")}))


if __name__=="__main__":main()
