"""Independent signed bit decoder covers all permitted widths and malformed tails."""
import unittest
from tools import rank_two_live_eight_mechanical_v2 as mechanical
from tools import rank_two_live_eight_model_v1 as model


class IndependentBits(unittest.TestCase):
    def test_signed_extremes_and_cross_byte_boundaries(self):
        for width in (4,6,7):
            limit=(1<<(width-1))-1
            values=[-limit,0,limit,-1,1,-limit,limit]
            self.assertEqual(mechanical.independent_codes(model.pack(values,width),len(values),width),values)

    def test_reserved_code_padding_and_dimensions_are_rejected(self):
        for width in (4,6,7):
            for payload,count in ((bytes([1<<(width-1)]),1),(bytes([128]),1),(b'',1)):
                with self.assertRaises(ValueError):mechanical.independent_codes(payload,count,width)


if __name__=='__main__':unittest.main()
