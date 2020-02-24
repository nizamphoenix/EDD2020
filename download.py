#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 02:21:58 2020

@author: https://github.com/sharibox/
"""
#!pip install boto3 (run this in your terminal/ ipython console)
#s3://ieee-dataport/competition/6486/EDD2020_release-I_2020-01-15_v2.zip

import boto3
s3 = boto3.client('s3')

#==========Communicate licence for dataset==========
print('Data you are downloading is protected under: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)')
print('see: https://creativecommons.org/licenses/by-nc/4.0/')
#===================================================
print('downloading... wait!!!')
s3 = boto3.client('s3', aws_access_key_id='AKIAIKFHV2ZSST7NLSEA' , aws_secret_access_key='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')
s3.download_file('ieee-dataport', 'competition/6486/EDD2020_release-I_2020-01-15_v2.zip', 'EDD2020.zip')
print('done... good luck with the EDD2020 challenge!!!')
#===================================================
#unzipping the data into a folder called EDD2020 in the current directory
import zipfile
with zipfile.ZipFile('EDD2020.zip', 'r') as zip_ref:
    zip_ref.extractall('EDD2020')
!ls ./EDD2020/EDD2020_release-I_2020-01-15
