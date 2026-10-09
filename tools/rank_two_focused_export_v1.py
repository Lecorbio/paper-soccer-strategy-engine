"""Exact-tensor exports with reversible C++ token and payload compaction."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import compact_representation as codec
from tools import rank_two_design_network_v1 as prior
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_minify as tokens
from tools import rank_two_network32_v2 as architecture_support

SCHEMA = "papersoccer.focused-network.runtime.v1"
TEMPLATE_PATH = campaign.ROOT / "submissions/codingame/research/focused-network/template.cpp"
PROFILES = {**campaign.FAMILIES, "control": [6301, 12, 8, 1]}
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"

B85_DECODER = r'''inline std::vector<std::uint8_t> decode_base85(std::string_view encoded,std::size_t expected){
constexpr std::string_view alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
if(!expected||encoded.size()!=((expected+3)/4)*5)throw std::invalid_argument("base85 length");
std::vector<std::uint8_t> bytes;bytes.reserve((encoded.size()/5)*4);
for(std::size_t offset=0;offset<encoded.size();offset+=5){
std::uint64_t value=0;
for(unsigned i=0;i<5;++i){const auto digit=alphabet.find(encoded[offset+i]);
if(digit==std::string_view::npos)throw std::invalid_argument("base85 digit");value=value*85+digit;}
if(value>UINT32_MAX)throw std::invalid_argument("base85 overflow");
for(int shift=24;shift>=0;shift-=8)bytes.push_back(static_cast<std::uint8_t>(value>>shift));}
for(std::size_t i=expected;i<bytes.size();++i)if(bytes[i])throw std::invalid_argument("base85 padding");
bytes.resize(expected);return bytes;
}
'''


def decode85(text, length):
    if length <= 0 or len(text) != ((length+3)//4)*5:
        raise ValueError("base85 length")
    result = bytearray()
    for offset in range(0, len(text), 5):
        value = 0
        for symbol in text[offset:offset+5]:
            if symbol not in ALPHABET:
                raise ValueError("base85 digit")
            value = value*85 + ALPHABET.index(symbol)
        if value > 0xffffffff:
            raise ValueError("base85 overflow")
        result.extend(value.to_bytes(4, "big"))
    if any(result[length:]):
        raise ValueError("base85 nonzero padding")
    payload = bytes(result[:length])
    if base64.b85encode(payload, pad=True).decode("ascii") != text:
        raise ValueError("base85 noncanonical text")
    return payload


def document(profile, values, scales, lineage=None):
    if profile not in PROFILES:
        raise ValueError("unknown profile")
    inputs, first, second, _ = PROFILES[profile]
    if len(values) != inputs*first + first*second + second:
        raise ValueError("weight dimensions")
    if len(scales) != 3 or any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0 for x in scales):
        raise ValueError("three finite positive layer scales required")
    packed = codec.pack(values, 4)
    body = dict(schema=SCHEMA, profile=profile, architecture=PROFILES[profile],
                feature_schema=architecture_support.FEATURES["distance-degree"][1],
                activations=["square-leaky-0.01", "leaky-relu-0.01", "fast-tanh-rational-v1"],
                biases=False, weight_bits=4, scales=[codec.f32(x) for x in scales],
                payload_base64=base64.b64encode(packed).decode(),
                payload_sha256=hashlib.sha256(packed).hexdigest(), lineage=lineage or {})
    return {**body, "body_sha256": hashlib.sha256(campaign.canonical(body)).hexdigest()}


def validate(runtime):
    if runtime.get("schema") != SCHEMA or runtime.get("profile") not in PROFILES:
        raise ValueError("runtime schema/profile")
    shape = PROFILES[runtime["profile"]]
    packed = base64.b64decode(runtime["payload_base64"], validate=True)
    values = codec.unpack(packed, shape[0]*shape[1]+shape[1]*shape[2]+shape[2],4)
    expected = document(runtime["profile"], values, runtime["scales"], runtime["lineage"])
    if expected != runtime:
        raise ValueError("runtime shape, activation, scale or payload binding differs")
    return values, packed


def minimal_whitespace(source):
    output = []
    previous = None
    for token in tokens.lex(source):
        if token.kind in ("space", "comment"):
            continue
        if token.kind == "directive":
            if output and not output[-1].endswith("\n"):
                output.append("\n")
            output.append(token.text if token.text.endswith("\n") else token.text+"\n")
            previous = None
            continue
        if previous is not None:
            pair = [(previous.kind,previous.text),(token.kind,token.text)]
            unsafe_literal = (previous.kind == "literal" and token.kind in ("word","number") or
                              token.kind == "literal" and previous.kind in ("word","number"))
            if unsafe_literal or tokens.signature(previous.text+token.text) != pair:
                output.append(" ")
        output.append(token.text)
        previous = token
    result = "".join(output)+"\n"
    if tokens.signature(result) != tokens.signature(source):
        raise ValueError("whitespace changed lexical signature")
    return result


def macro_names(compiler="/usr/bin/clang++", flags=()):
    source = TEMPLATE_PATH.read_text()
    includes = "\n".join(line for line in source.splitlines() if line.startswith("#include"))+"\n"
    result = subprocess.run([compiler,"-std=c++20",*flags,"-dM","-E","-x","c++","-"],
                            input=includes,text=True,capture_output=True,timeout=60,check=True)
    return set(re.findall(r"^#define\s+(\w+)",result.stdout,re.M))


def alias_compact(source, forbidden=()):
    items = list(tokens.lex(source))
    identifiers = {t.text for t in items if t.kind == "word"} | set(forbidden)
    # Macro/directive references retain their spelling. Exclude all such words.
    excluded = {word for t in items if t.kind == "directive" for word in re.findall(r"\b\w+\b",t.text)}
    counts = Counter(t.text for t in items if t.kind == "word" and t.text not in excluded)
    candidates = sorted(counts,key=lambda word:(-counts[word]*max(len(word)-2,0),word))
    aliases = {}
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    import itertools
    available = itertools.chain(iter(alphabet), (a+b for a in alphabet for b in alphabet))
    for word in candidates:
        alias = next((a for a in available if a not in identifiers and a not in excluded),None)
        if alias is None:
            break
        definition = f"#define {alias} {word}\n"
        if counts[word]*(len(word)-len(alias)) <= len(definition):
            continue
        aliases[word] = alias
        identifiers.add(alias)
    if not aliases:
        return minimal_whitespace(source), {}
    replaced = "".join(aliases.get(t.text,t.text) if t.kind == "word" else t.text for t in items)
    # System headers are fully read before project aliases become active.
    last_include = max(t.end for t in items if t.kind == "directive" and t.text.startswith("#include"))
    prefix = "".join(t.text for t in items if t.end <= last_include)
    suffix = "".join(aliases.get(t.text,t.text) if t.kind == "word" else t.text
                     for t in items if t.start >= last_include)
    definitions = "".join(f"#define {alias} {word}\n" for word,alias in aliases.items())
    # A submitted source is a complete standalone translation unit. Native
    # include/probe adapters must append alias_cleanup() before their own code.
    compact = minimal_whitespace(prefix+definitions+suffix)
    return compact, aliases


def alias_cleanup(aliases):
    return "\n"+"".join(f"#undef {alias}\n" for alias in aliases.values())


def compiler_tokens(source, compiler="/usr/bin/clang++", flags=()):
    result = subprocess.run([compiler,"-std=c++20",*flags,"-Xclang","-dump-tokens","-fsyntax-only","-x","c++","-"],
                            input=source,text=True,capture_output=True,timeout=90)
    if result.returncode:
        raise ValueError("compiler token oracle failed: "+result.stderr[-2000:])
    parsed = []
    for line in result.stderr.splitlines():
        match = re.match(r"^(\w+) '(.*)'\s+.*Loc=<",line)
        if match:
            parsed.append((match[1],match[2]))
    if len(parsed)<100:
        raise ValueError("compiler token oracle returned no substantial token stream")
    return parsed


def verify_compaction(original, compact, compiler="/usr/bin/clang++", flags=()):
    before,after = compiler_tokens(original,compiler,flags),compiler_tokens(compact,compiler,flags)
    if before != after:
        index = next((i for i,(a,b) in enumerate(zip(before,after)) if a != b),min(len(before),len(after)))
        raise ValueError(f"compiler token stream differs at {index}: {before[index:index+2]} / {after[index:index+2]}")
    return dict(passed=True,tokens=len(before),sha256=hashlib.sha256(repr(before).encode()).hexdigest(),flags=list(flags))


def uncompressed_source(runtime, kind):
    values,packed = validate(runtime)
    template = TEMPLATE_PATH.read_bytes()
    if hashlib.sha256(template).hexdigest() != prior.TEMPLATE_SHA256:
        raise ValueError("bound verified search template changed")
    source = template.decode("ascii")
    inputs,first,second,_ = runtime["architecture"]
    compressed,lengths,_ = codec.compress(values,4)
    if codec.decompress(compressed,len(values),4,lengths) != values:
        raise ValueError("Huffman roundtrip")
    payload = compressed if kind == "huffman" else packed
    if kind not in ("huffman","packed"):
        raise ValueError("payload packing kind")
    text = base64.b85encode(payload,pad=True).decode("ascii")
    if decode85(text,len(payload)) != payload:
        raise ValueError("independent transport roundtrip")
    for key,value in {"kInputs":inputs,"kHiddenOne":first,"kHiddenTwo":second,
                      "kWeightCount":len(values),"kPackedByteCount":len(packed),
                      "kFirstSearchTimeMs":550,"kLaterSearchTimeMs":140}.items():
        source = prior._constant(source,key,str(value))
    for key,value in zip(("kScaleOne","kScaleTwo","kScaleThree"),runtime["scales"]):
        literal=format(value,".9g")
        if not any(c in literal for c in ".eE"): literal += ".0"
        source=prior._constant(source,key,literal+"F")
    for key,value in {"kRuntimeSchema":SCHEMA,"kFeatureSchema":runtime["feature_schema"],
                      "kPayloadSha256":runtime["payload_sha256"],"kRuntimeBodySha256":runtime["body_sha256"],
                      "kIdentity":"focused-"+runtime["profile"]+"-"+runtime["body_sha256"][:12]}.items():
        source=prior._constant(source,key,json.dumps(value),string=True)
    source,count=re.subn(r"kHuffmanLengths\{[^}]+\}","kHuffmanLengths{"+",".join(map(str,lengths))+"}",source)
    if count != 1: raise ValueError("Huffman header anchor")
    match = list(architecture_support.PAYLOAD.finditer(source))
    if len(match)!=1: raise ValueError("single payload literal required")
    match=match[0]
    literals='"'+text+'"'
    source=source[:match.start(1)]+"\n"+literals+source[match.end(1):]
    source=prior._replace(source,"inline constexpr bool kBootstrapZero = false;",
                          "inline constexpr bool kBootstrapZero = false;\ninline constexpr std::size_t kEncodedByteCount="+str(len(payload))+";")
    start=source.index("inline int base64_value(")
    end=source.index("inline const std::vector<std::int8_t> &weights()",start)
    source=source[:start]+B85_DECODER+source[end:]
    source=prior._replace(source,"const auto compressed = decode_base64(learned_model::kPackedWeights);",
                          "const auto compressed=decode_base85(learned_model::kPackedWeights,learned_model::kEncodedByteCount);")
    source=prior._replace(source,"const auto packed = learned_codec::decode(compressed, learned_model::kWeightCount,\nlearned_model::kWeightBits, learned_model::kHuffmanLengths);",
                          "const auto packed="+("learned_codec::decode(compressed,learned_model::kWeightCount,learned_model::kWeightBits,learned_model::kHuffmanLengths);" if kind=="huffman" else "compressed;"))
    source=prior._replace(source,"static_assert(learned_model::kInputs == 6301 && learned_model::kHiddenOne == 12 &&\nlearned_model::kHiddenTwo == 8 && learned_model::kWeightBits == 4);",
                          f"static_assert(learned_model::kInputs=={inputs}&&learned_model::kHiddenOne=={first}&&learned_model::kHiddenTwo=={second}&&learned_model::kWeightBits==4);")
    start=source.index("inline float evaluate(const Features &features) {")
    end=source.index("\n}   ",start)
    source=prior._replace(source,source[start:end],architecture_support.EVALUATE)
    # Search ends before the common response envelope to leave cleanup/output time.
    source=prior._replace(source,"action = choose_complete_turn(state, response_deadline);",
                          "action=choose_complete_turn(state,response_deadline-std::chrono::milliseconds(10));")
    diagnostics = {}
    pattern = r'(throw\s+std::(?:invalid_argument|logic_error|runtime_error|out_of_range|length_error)\s*\(\s*)("(?:[^"\\]|\\.)*")(\s*\))'
    def diagnostic(match):
        message = json.loads(match[2])
        code = "e" + str(len(diagnostics))
        diagnostics[code] = message
        return match[1] + json.dumps(code) + match[3]
    source = re.sub(pattern, diagnostic, source)
    if kind == "packed":
        start = source.index("namespace papersoccer::turn_action_v2::learned_codec {")
        end = source.index("namespace papersoccer::turn_action_v2::learned_model {", start)
        source = source[:start] + source[end:]
    return source,text,payload,diagnostics


def export(runtime, forbidden=(), verify_tokens=False, compiler="/usr/bin/clang++"):
    options=[]
    for kind in ("packed","huffman"):
        original,text,payload,diagnostics=uncompressed_source(runtime,kind)
        compact,aliases=alias_compact(original,forbidden)
        options.append((len(compact),kind,compact,original,text,payload,aliases,diagnostics))
    _,kind,source,original,text,payload,aliases,diagnostics=min(options,key=lambda row:(row[0],row[1]))
    proofs=[]
    if verify_tokens:
        for flags in ((),("-DPAPER_SOCCER_TURN_ACTION_V2_NO_MAIN",),
                      ("-U__clang__",),("-U__clang__","-DPAPER_SOCCER_TURN_ACTION_V2_NO_MAIN")):
            proofs.append(verify_compaction(original,source,compiler,flags))
    report=dict(schema=SCHEMA+".export",source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                characters=len(source),nonpayload_characters=len(source)-len(text),
                below_hard_limit=len(source)<100000,below_stress_limit=len(source)<98000,
                nonpayload_target_passed=len(source)-len(text)<=43000,
                payload_kind=kind,transport="python-base85-big-endian-padded-v1",
                transport_bytes=len(payload),tensor_sha256=runtime["payload_sha256"],
                runtime_sha256=runtime["body_sha256"],architecture=runtime["architecture"],
                aliases=aliases,compiler_token_proofs=proofs,clocks_ms=[550,140],
                diagnostic_messages=diagnostics,probe_requires_alias_cleanup=True,
                provisional_cleanup_reserve_ms=10,whole_response_timing_certified=False,
                native_parity_verified=False,live_admitted=False,
                options=[dict(kind=row[1],characters=row[0]) for row in options])
    return source,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--verify-tokens",action="store_true")
    args=parser.parse_args()
    source,report=export(campaign.read(args.runtime),macro_names(),args.verify_tokens)
    args.output.mkdir(parents=True,exist_ok=True)
    campaign.immutable(args.output/"submission.cpp",source.encode())
    campaign.immutable(args.output/"EXPORT.json",report)
    print(json.dumps({k:report[k] for k in ("characters","nonpayload_characters","payload_kind","below_hard_limit","nonpayload_target_passed")}))


if __name__=="__main__":
    main()
