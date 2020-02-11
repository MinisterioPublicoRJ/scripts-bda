#!/bin/sh

spark2-submit --py-files packages/*.whl,packages/*.egg tabela_distribuicao.py
