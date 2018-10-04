#!/usr/bin/env python

import subprocess
import unittest
import glob
import os
from io import StringIO

script = "../src/skyline_converter.py"
out_file_conv = "out_files/skyline_from_xtract.ssl"
in_file_conv = "in_files/skyline_xtract_input.csv"
exp_file_conv = "exp_files/skyline_from_xtract.ssl"

out_file_quant = "out_files/skyline_quant_results.csv"
in_file_quant_xtract = "in_files/skyline_quant_input.ssl"
in_file_quant_skyline = "in_files/skyline_quant_output.csv"
exp_file_quant = "exp_files/skyline_quant_results.csv"

def test_quant():
    out = subprocess.check_output(["python3", script, "-ix", in_file_quant_xtract,
                                   "-is", in_file_quant_skyline, "-er", "Inact", "-o", out_file_quant])
    print_output(out)


def test_converter():
    out = subprocess.check_output(["python3", script, "-ix", in_file_conv, "-o", out_file_conv])
    print_output(out)

def print_output(out):
    for line in out.splitlines():
        print(line)


class FunctionTest(unittest.TestCase):
    def test_converter_output(self):
        test_converter()
        with open(out_file_conv, 'r') as o_f:
            o_cont = o_f.read()
            with open(exp_file_conv, 'r') as e_f:
                e_cont = e_f.read()
                self.assertEqual(o_cont, e_cont)

    def test_quant_output(self):
        test_quant()
        with open(out_file_quant, 'r') as o_f:
            o_cont = o_f.read()
            with open(exp_file_quant, 'r') as e_f:
                e_cont = e_f.read()
                self.assertEqual(o_cont, e_cont)

def main():
    merger_test = FunctionTest()
    merger_test.test_converter_output()
    merger_test.test_quant_output()

if __name__ == '__main__':
    main()
