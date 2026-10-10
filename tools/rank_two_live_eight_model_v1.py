"""Exact 4/6/7-bit 8/8 tensors and standalone source exports."""
from __future__ import annotations

import base64
import hashlib
import math
import re
from tools import rank_two_focused_export_v2 as retained
from tools import rank_two_live_eight_v1 as campaign

SCHEMA = "papersoccer.live-eight.runtime.v1"
SHAPES = {"dd8": [6301, 8, 8, 1], "control": [6301, 12, 8, 1]}


def pack(values, bits):
    if bits not in (4, 6, 7):
        raise ValueError("unsupported precision")
    maximum = (1 << (bits-1))-1
    result = bytearray((len(values)*bits+7)//8)
    for index, value in enumerate(values):
        if type(value) is not int or not -maximum <= value <= maximum:
            raise ValueError("reserved or noninteger tensor code")
        bit = index*bits
        code = value & ((1 << bits)-1)
        result[bit//8] |= (code << (bit%8)) & 255
        if bit%8+bits > 8:
            result[bit//8+1] |= code >> (8-bit%8)
    return bytes(result)


def unpack(payload, count, bits):
    if bits not in (4, 6, 7) or type(count) is not int or count < 0 or len(payload) != (count*bits+7)//8:
        raise ValueError("tensor payload dimensions")
    if count*bits%8 and payload and payload[-1] >> (count*bits%8):
        raise ValueError("nonzero tensor padding")
    values=[]; sign=1 << (bits-1)
    for index in range(count):
        bit=index*bits;word=payload[bit//8]
        if bit//8+1<len(payload):word |= payload[bit//8+1] << 8
        code=(word >> (bit%8)) & ((1 << bits)-1)
        if code == sign:raise ValueError("reserved negative tensor code")
        values.append(code-(1 << bits) if code & sign else code)
    return values


def document(profile, values, scales, bits=4, lineage=None):
    if profile not in SHAPES or (profile == "control" and bits != 4):
        raise ValueError("declared 8/8 or exact incumbent control required")
    shape=SHAPES[profile];count=shape[0]*shape[1]+shape[1]*shape[2]+shape[2]
    if len(values)!=count or len(scales)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in scales):
        raise ValueError("tensor shape or scales")
    payload=pack(values,bits)
    body=dict(schema=SCHEMA,profile=profile,architecture=shape,feature_schema=campaign.FEATURE_SCHEMA,
              activations=campaign.ACTIVATIONS,biases=False,weight_bits=bits,
              scales=[retained.codec.f32(x) for x in scales],payload_base64=base64.b64encode(payload).decode(),
              payload_sha256=hashlib.sha256(payload).hexdigest(),lineage=lineage or {})
    return {**body,"body_sha256":hashlib.sha256(campaign.old.retained.canonical(body)).hexdigest()}


def validate(runtime):
    if runtime.get("schema")!=SCHEMA or runtime.get("profile") not in SHAPES:
        raise ValueError("runtime schema/profile")
    shape=SHAPES[runtime["profile"]];count=shape[0]*shape[1]+shape[1]*shape[2]+shape[2]
    payload=base64.b64decode(runtime["payload_base64"],validate=True)
    values=unpack(payload,count,runtime["weight_bits"])
    if runtime != document(runtime["profile"],values,runtime["scales"],runtime["weight_bits"],runtime["lineage"]):
        raise ValueError("runtime tensor/activation identity differs")
    return values,payload


def uncompressed_source(runtime, clocks):
    if list(clocks) not in campaign.PLAN["clock_ladder"]:
        raise ValueError("undeclared response tuple")
    values,payload=validate(runtime)
    dummy=retained.document(runtime["profile"],[0]*len(values),runtime["scales"])
    source,old_text,_,diagnostics=retained.uncompressed_source(dummy,"packed")
    text=base64.b85encode(payload,pad=True).decode("ascii")
    if retained.decode85(text,len(payload))!=payload:
        raise ValueError("independent transport parity")
    source=source.replace('"'+old_text+'"','"'+text+'"')
    for key,value in dict(kWeightBits=runtime["weight_bits"],kPackedByteCount=len(payload),
                          kEncodedByteCount=len(payload),kFirstSearchTimeMs=clocks[0],kLaterSearchTimeMs=clocks[1]).items():
        pattern=r'(inline constexpr (?:unsigned|std::size_t) '+key+r'\s*=\s*)[^;]+;'
        if key.startswith('kFirst') or key.startswith('kLater'):
            pattern=r'(constexpr std::uint32_t '+key+r'\s*=\s*)[^;]+;'
        source,count=re.subn(pattern,lambda m:m[1]+str(value)+';',source)
        if count!=1:raise ValueError("source constant anchor: "+key)
    source=source.replace('learned_model::kWeightBits==4','learned_model::kWeightBits=='+str(runtime["weight_bits"]))
    start=source.index("std::vector<std::int8_t> decoded(learned_model::kWeightCount);")
    end=source.index("return decoded;",start)
    decoder='''std::vector<std::int8_t> decoded(learned_model::kWeightCount);
constexpr unsigned width=learned_model::kWeightBits,sign=1U<<(width-1);
for(std::size_t i=0;i<decoded.size();++i){
const auto bit=i*width;unsigned word=packed[bit/8];
if(bit/8+1<packed.size())word|=unsigned(packed[bit/8+1])<<8;
const unsigned code=(word>>(bit%8))&((1U<<width)-1);
if(code==sign)throw std::invalid_argument("reserved tensor code");
decoded[i]=static_cast<std::int8_t>(code&sign?int(code)-int(1U<<width):int(code));}
'''
    source=source[:start]+decoder+source[end:]
    source=source.replace(dummy["body_sha256"],runtime["body_sha256"])
    return source,text,diagnostics


def export(runtime, clocks=(550,140), verify_tokens=False, compiler="/usr/bin/clang++"):
    original,text,diagnostics=uncompressed_source(runtime,clocks)
    source,aliases=retained.alias_compact(original,retained.macro_names(compiler))
    proofs=[]
    if verify_tokens:
        for flags in ((),("-DPAPER_SOCCER_TURN_ACTION_V2_NO_MAIN",),("-U__clang__",),
                      ("-U__clang__","-DPAPER_SOCCER_TURN_ACTION_V2_NO_MAIN")):
            proofs.append(retained.verify_compaction(original,source,compiler,flags))
    if len(source)>99000:raise ValueError("source exceeds declared 99,000-character target")
    return source,dict(schema=SCHEMA+".export",source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        characters=len(source),nonpayload_characters=len(source)-len(text),aliases=aliases,
        compiler_token_proofs=proofs,payload_kind="packed",weight_bits=runtime["weight_bits"],
        tensor_sha256=runtime["payload_sha256"],architecture=runtime["architecture"],clocks_ms=list(clocks),
        diagnostics=diagnostics,whole_response_cleanup_reserve_ms=10,
        timing_certified=False,native_parity_verified=False,live_admitted=False)


def quantize(parameters, scales, bits):
    from tools import rank_two_focused_training_v9 as training
    if bits not in (4,6,7) or set(scales)!=set(parameters) or any(not math.isfinite(float(x)) or x<=0 for x in scales.values()):
        raise ValueError("precision or finite layer scales differ")
    np=training.np;maximum=(1 << (bits-1))-1
    integer={key:np.clip(np.rint(value/np.float32(scales[key])),-maximum,maximum).astype(np.int8) for key,value in parameters.items()}
    return training.core.QuantizedWeights(integer,{key:np.float32(value) for key,value in scales.items()})
