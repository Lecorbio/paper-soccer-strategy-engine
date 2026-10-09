"""Independent payload transport and lexical compaction boundary checks."""
import base64
import unittest
from tools import rank_two_focused_export_v1 as exporter


class ExportTests(unittest.TestCase):
    def test_base85_padding_and_independent_roundtrip(self):
        for length in range(1,34):
            raw=bytes(range(length));text=base64.b85encode(raw,pad=True).decode()
            self.assertEqual(exporter.decode85(text,length),raw)

    def test_base85_bad_lengths_digits_overflow_and_tail(self):
        for text,length in (("",1),("~~~~~",4),("0000\"",4),
                            (base64.b85encode(b"abcd",pad=True).decode(),1)):
            with self.subTest(text=text),self.assertRaises(ValueError):exporter.decode85(text,length)

    def test_directives_literals_and_include_adapter(self):
        source='#include <string>\nstd::string destination="destination";\nint main(){return destination.size()+destination.size()+destination.size();}\n'
        compact,aliases=exporter.alias_compact(source)
        self.assertIn('"destination"',compact)
        self.assertTrue(compact.startswith('#include <string>\n'))
        self.assertIn("destination",aliases)
        cleanup=exporter.alias_cleanup(aliases)
        self.assertIn("#undef "+aliases["destination"],cleanup)

    def test_reserved_quantized_code_rejected(self):
        shape=exporter.PROFILES["dd8"];count=shape[0]*shape[1]+shape[1]*shape[2]+shape[2]
        with self.assertRaises(ValueError):exporter.document("dd8",[-8]*count,[.01,.01,.01])


if __name__=="__main__":unittest.main()
