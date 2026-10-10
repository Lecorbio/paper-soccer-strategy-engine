"""Independent bit-stream and native evaluator checks for higher precision."""
import random
from pathlib import Path
import tempfile
import subprocess
import unittest
import numpy as np
from tools import rank_two_live_eight_model_v1 as model
from tools import rank_two_focused_native_v4 as native
from tools import rank_two_focused_training_v9 as training


class Precision(unittest.TestCase):
    def test_native_adapter_keeps_disabled_platform_headers_disabled(self):
        source = '#if 0\n#include <unavailable-platform-header.h>\n#endif\nnamespace fixture {}\n'
        probe = native.probe_source(source, {})
        result = subprocess.run(['/usr/bin/clang++', '-std=c++20', '-E', '-x', 'c++', '-'],
                                input=probe, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_independent_signed_bit_stream_and_padding(self):
        rng=random.Random(2026101001)
        for width in (4,6,7):
            maximum=(1 << (width-1))-1
            for count in (1,2,7,19,83):
                values=[rng.randint(-maximum,maximum) for _ in range(count)]
                bits=''.join(''.join(str((x & ((1<<width)-1))>>bit&1) for bit in range(width)) for x in values)
                bits=bits.ljust((len(bits)+7)//8*8,'0')
                expected=bytes(sum(int(bits[start+i])<<i for i in range(8)) for start in range(0,len(bits),8))
                self.assertEqual(model.pack(values,width),expected)
                self.assertEqual(model.unpack(expected,count,width),values)
            with self.assertRaises(ValueError):model.pack([-(1 << (width-1))],width)
            with self.assertRaises(ValueError):model.unpack(bytes([1 << (width-1)]),1,width)
            with self.assertRaises(ValueError):model.unpack(bytes([255]),1,width)

    def test_runtime_tampering_and_control_width(self):
        values=[0]*50480;r=model.document('dd8',values,[.01]*3,6)
        self.assertEqual(model.validate(r)[0],values)
        with self.assertRaises(ValueError):model.validate({**r,'activations':['relu']})
        with self.assertRaises(ValueError):model.document('control',[0]*75716,[.01]*3,6)

    def test_native_six_bit_tensor_and_float_parity(self):
        width=6;rng=random.Random(2026101002);values=[rng.randint(-31,31) for _ in range(50480)]
        scales=[.001,.002,.003];r=model.document('dd8',values,scales,width)
        # Token-oracle admission is separately required for production exports.
        source,report=model.export(r,verify_tokens=False)
        self.assertLessEqual(len(source),99000)
        with tempfile.TemporaryDirectory(prefix='live-eight-native-') as directory:
            binary=native.build(source,report,Path(directory))
            answers=native.query(binary,['weights','eval 3 0 316 6300'])
        self.assertEqual([int(x) for x in answers[0].split()],values)
        architecture,parameters=training.initialize('dd8',0)
        offset=0;integers={}
        for key,shape in architecture.shapes.items():
            count=int(np.prod(shape));integers[key]=np.asarray(values[offset:offset+count],dtype=np.int8).reshape(shape)
            parameters[key][:]=integers[key]*np.float32(scales[len(integers)-1]);offset+=count
        q=training.core.QuantizedWeights(integers,{key:np.float32(scale) for key,scale in zip(integers,scales)})
        expected=training.forward(parameters,architecture,[[0,316,6300]],q)[0][0]
        self.assertEqual(np.float32(answers[1]),expected)


if __name__=='__main__':unittest.main()
