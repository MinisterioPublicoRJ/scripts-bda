import requests
import subprocess
import array
import time
import argparse
import asyncio

import aiohttp
from aiohttp import ClientSession
from tqdm import tqdm


async def call_endpoint(url, i, response_times, session):
    print(f"Calling {url}")
    t1 = time.time()
    response = await session.request(method="GET", url=url)
    t2 = time.time()

    if response.status == 200:
        print(t2-t1, url)
        response_times[i].append(t2 - t1)

async def simulate_one_client(urls, response_times):
    tasks = []
    async with ClientSession() as session:
        for i, url in enumerate(urls):
            tasks.append(call_endpoint(url, i, response_times, session))

        await asyncio.wait(tasks)

async def simulate_n_clients(response_times, endpoints, token, n=10):
    with open('ids.csv', 'r') as ids:
        lines = ids.readlines()[1:]
    
    for i in range(0, len(lines), n):
        tasks = []
        for line in lines[i:i+n]:
            orgao_id = line.split(',')[0]
            cpf = line.split(',')[1][:-1]
            
            urls = build_urls(endpoints, orgao_id, cpf, token)
            tasks.append(simulate_one_client(urls, response_times))

        await asyncio.wait(tasks)

def build_urls(endpoints, orgao, cpf, token):
    return [url.format(orgao=orgao, cpf=cpf, token=token) for url in endpoints]


parser = argparse.ArgumentParser(description="Endpoints Speed Test")
parser.add_argument('-p','--parallel', dest='runParallel', action='store_true')
parser.set_defaults(runParallel=False)
args = parser.parse_args()

RUN_PARALLEL = args.runParallel

endpoints = [
    'https://d-apimpmapas.mprj.mp.br/dominio/saidas/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/outliers/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/entradas/{orgao}/{cpf}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/suamesa/detalhe/vistas/{orgao}/{cpf}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/suamesa/vistas/{orgao}/{cpf}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/suamesa/finalizados/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/suamesa/investigacoes/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/suamesa/processos/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/alertas/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/radar/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/suamesa/lista/vistas/{orgao}/{cpf}/ate_vinte?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/suamesa/detalhe/processos/{orgao}?jwt={token}​',
    'https://d-apimpmapas.mprj.mp.br/dominio/suamesa/detalhe/investigacoes/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/lista/processos/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/tempo-tramitacao/{orgao}?jwt={token}',
    'https://d-apimpmapas.mprj.mp.br/dominio/desarquivamentos/{orgao}?jwt={token}',
]

response_times = [[]] * len(endpoints)

token = subprocess.run(["curl -s -X POST "
"-F 'jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX25hbWUiOiJTSURORVlST1NBIiwic2NvcGUiOlsibGVpdHVyYSIsImVzY3JpdGEiXSwic2NhVXNlciI6eyJob3N0IjpudWxsLCJjcGZVc3VhcmlvIjoiMDczNTUwMDU3NjYiLCJsb2dpblVzdWFyaW8iOiJTSURORVlST1NBIiwibm9tZVVzdWFyaW8iOiJTSURORVkgUk9TQSBEQSBTSUxWQSBKVU5JT1IiLCJvcmdhb1VzdWFyaW8iOjQwMDU1MSwibm9tZU9yZ2FvVXN1YXJpbyI6IjE5wqogUFJPTU9UT1JJQSBERSBKVVNUScOHQSBERSBSRUdJw4NPIEVTUEVDSUFMIiwic2lnbGFPcmdhb1VzdWFyaW8iOiJQSjE5UkVTUCIsImlkIjoiZDNhMjEzYTdmYTZjZGFiNzVlYzkwYTE2NmNjOWJkMzM0MjFjNWQyYzQzZjUzMDQzZGFlZGIwOTAxZGQ5NTc2ZSIsImlwIjpudWxsLCJzaXN0ZW1hIjozNTQ4Niwibm9tZVNpc3RlbWEiOiJNR1BFIiwicm9sZXNTaXN0ZW1hIjpbXSwib3JnYW8iOjQwMDU1MSwibm9tZU9yZ2FvIjoiTUlOSVNUw4lSSU8gUMOaQkxJQ08gRE8gRVNUQURPIERPIFJJTyBERSBKQU5FSVJPIiwic2lnbGFPcmdhbyI6bnVsbCwicGVyZmlsIjo1ODAzNiwibm9tZVBlcmZpbCI6ImFjZXNzbyIsImNsYXNzZSI6Mzg0Niwibm9tZUNsYXNzZSI6Ik1FTUJSTyBBVElWTyIsIm9yaWdlbSI6bnVsbCwiY29tcG9uZW50ZXMiOnt9LCJmdW5jaW9uYWxpZGFkZXMiOlsiY29uc3VsdGFyX2ludGltYWNvZXNfcHJpbWVpcmFfaW5zdGFuY2lhIiwianVkaWNpYWxfcmVub21lYXJfdGl0dWxvX2FsdGVybmF0aXZvIiwiYWNlc3NvIiwiaW50ZWdyYV9qdWRpY2lhbF9wZXRpY2lvbmFtZW50byIsImdlc3Rhb19wcmltZWlyYV9pbnN0YW5jaWEiLCJhY2Vzc29fanVkaWNpYWxfcG9ydGFsIiwiaW50ZWdyYV9qdWRpY2lhbF92aXN1YWxpemFyX3BlY2FzX3Byb2MiLCJpbnRlZ3JhX2p1ZGljaWFsX2Rpc3RyaWJ1aWNhb19waXAiLCJqdWRpY2lhbF9tb2RlbG9zX3BldGljaW9uYW1lbnRvcyIsIm1ncGVfaXAiLCJpbnRlZ3JhX2p1ZGljaWFsX2FiZXJ0dXJhIiwianVkaWNpYWxfbWFudGVyX21pbnV0YV9pbnRpbWFjYW8iLCJpbnF1ZXJpdG9fZW52aWFyX3Byb21vY2FvIiwiaW50ZWdyYV9qdWRpY2lhbF9wZXRpY2lvbmFtZW50b19wb3JfY290YSIsImlucXVlcml0b19jcmltaW5hbCIsImlucXVlcml0b19kaXN0cmlidWlyIiwiaW50ZWdyYV9qdWRpY2lhbF9yZW5vbWVhcl9pbmRpY2UiXSwidGlja2V0IjoiY2U1ZTJkMTFjY2M1NGEzZTljNWExNjk5NDA3ZGNkYjY1OTM4MDg5NTY1MTAwMCIsInBhcmFtZXRyb3MiOnsic3NvX2hhYmlsaXRhZG8iOiJ0cnVlIiwic3NvX3VybF9yZWRpcmVjaW9uYW1lbnRvIjoiaHR0cDovL2QtYXBwcy5tcHJqLm1wLmJyL3Npc3RlbWEvbWdwZS8jLyIsInNzb19kYXRhc291cmNlX2puZGlfbmFtZSI6ImphdmE6amJvc3MvbWdwZURTIn0sInBlc3NESyI6NTA1ODk1LCJtYXRyaWN1bGEiOiIwMDAwMzI0MCIsImdyYW50ZWRBdXRocyI6W3siYXV0aG9yaXR5IjoiUk9MRV9DT05TVUxUQVJfSU5USU1BQ09FU19QUklNRUlSQV9JTlNUQU5DSUEifSx7ImF1dGhvcml0eSI6IlJPTEVfSlVESUNJQUxfUkVOT01FQVJfVElUVUxPX0FMVEVSTkFUSVZPIn0seyJhdXRob3JpdHkiOiJST0xFX0FDRVNTTyJ9LHsiYXV0aG9yaXR5IjoiUk9MRV9JTlRFR1JBX0pVRElDSUFMX1BFVElDSU9OQU1FTlRPIn0seyJhdXRob3JpdHkiOiJST0xFX0dFU1RBT19QUklNRUlSQV9JTlNUQU5DSUEifSx7ImF1dGhvcml0eSI6IlJPTEVfQUNFU1NPX0pVRElDSUFMX1BPUlRBTCJ9LHsiYXV0aG9yaXR5IjoiUk9MRV9JTlRFR1JBX0pVRElDSUFMX1ZJU1VBTElaQVJfUEVDQVNfUFJPQyJ9LHsiYXV0aG9yaXR5IjoiUk9MRV9JTlRFR1JBX0pVRElDSUFMX0RJU1RSSUJVSUNBT19QSVAifSx7ImF1dGhvcml0eSI6IlJPTEVfSlVESUNJQUxfTU9ERUxPU19QRVRJQ0lPTkFNRU5UT1MifSx7ImF1dGhvcml0eSI6IlJPTEVfTUdQRV9JUCJ9LHsiYXV0aG9yaXR5IjoiUk9MRV9JTlRFR1JBX0pVRElDSUFMX0FCRVJUVVJBIn0seyJhdXRob3JpdHkiOiJST0xFX0pVRElDSUFMX01BTlRFUl9NSU5VVEFfSU5USU1BQ0FPIn0seyJhdXRob3JpdHkiOiJST0xFX0lOUVVFUklUT19FTlZJQVJfUFJPTU9DQU8ifSx7ImF1dGhvcml0eSI6IlJPTEVfSU5URUdSQV9KVURJQ0lBTF9QRVRJQ0lPTkFNRU5UT19QT1JfQ09UQSJ9LHsiYXV0aG9yaXR5IjoiUk9MRV9JTlFVRVJJVE9fQ1JJTUlOQUwifSx7ImF1dGhvcml0eSI6IlJPTEVfSU5RVUVSSVRPX0RJU1RSSUJVSVIifSx7ImF1dGhvcml0eSI6IlJPTEVfSU5URUdSQV9KVURJQ0lBTF9SRU5PTUVBUl9JTkRJQ0UifV0sImN1YSI6Ijk3NWYzZWE0LTY1MDUtNDg2MC1iYTlmLTViYWVkMWFkOTE3Ny1hYjc4IiwiYXV0aG9yaXRpZXMiOm51bGwsInBhc3N3b3JkIjpudWxsLCJ1c2VybmFtZSI6IlNJRE5FWVJPU0EiLCJhY2NvdW50Tm9uRXhwaXJlZCI6ZmFsc2UsImFjY291bnROb25Mb2NrZWQiOmZhbHNlLCJjcmVkZW50aWFsc05vbkV4cGlyZWQiOmZhbHNlLCJlbmFibGVkIjpmYWxzZSwiZGVzY3JpY2FvVXN1YXJpbyI6IlNJRE5FWSBST1NBIERBIFNJTFZBIEpVTklPUiAoU0lETkVZUk9TQSkiLCJlbXB0eSI6ZmFsc2V9LCJleHAiOjE1ODQ1NjAzODAsImF1dGhvcml0aWVzIjpbIlJPTEVfSU5URUdSQV9KVURJQ0lBTF9QRVRJQ0lPTkFNRU5UTyIsIlJPTEVfSU5URUdSQV9KVURJQ0lBTF9BQkVSVFVSQSIsIlJPTEVfSU5RVUVSSVRPX0VOVklBUl9QUk9NT0NBTyIsIlJPTEVfQUNFU1NPX0pVRElDSUFMX1BPUlRBTCIsIlJPTEVfSU5URUdSQV9KVURJQ0lBTF9ESVNUUklCVUlDQU9fUElQIiwiUk9MRV9JTlRFR1JBX0pVRElDSUFMX1JFTk9NRUFSX0lORElDRSIsIlJPTEVfQUNFU1NPIiwiUk9MRV9JTlRFR1JBX0pVRElDSUFMX1ZJU1VBTElaQVJfUEVDQVNfUFJPQyIsIlJPTEVfSU5URUdSQV9KVURJQ0lBTF9QRVRJQ0lPTkFNRU5UT19QT1JfQ09UQSIsIlJPTEVfQ09OU1VMVEFSX0lOVElNQUNPRVNfUFJJTUVJUkFfSU5TVEFOQ0lBIiwiUk9MRV9NR1BFX0lQIiwiUk9MRV9JTlFVRVJJVE9fRElTVFJJQlVJUiIsIlJPTEVfSlVESUNJQUxfTU9ERUxPU19QRVRJQ0lPTkFNRU5UT1MiLCJST0xFX0lOUVVFUklUT19DUklNSU5BTCIsIlJPTEVfSlVESUNJQUxfUkVOT01FQVJfVElUVUxPX0FMVEVSTkFUSVZPIiwiUk9MRV9HRVNUQU9fUFJJTUVJUkFfSU5TVEFOQ0lBIiwiUk9MRV9KVURJQ0lBTF9NQU5URVJfTUlOVVRBX0lOVElNQUNBTyJdLCJqdGkiOiJhMzI2ZjMzNi0xYzAwLTRkNjMtYjdiNC0zOTZhZDhjOWY5NDYiLCJjbGllbnRfaWQiOiJtZ3BlLWxvZ2luLWFwcCJ9.bGLaVp4HDZmu87A1M2o4_zCRfFc4Ucd8XLqnabBXnBo' "
"""https://d-apimpmapas.mprj.mp.br/dominio/token/login/ | jq .token | sed -e 's/"//g'"""], shell=True, capture_output=True).stdout
token = token.decode('utf-8')[:-1]

if RUN_PARALLEL:
    asyncio.run(simulate_n_clients(response_times, endpoints, token, n=10))
else:
    with open('ids.csv', 'r') as ids:
        for line in tqdm(ids.readlines()[1:]):
            orgao_id = line.split(',')[0]
            cpf = line.split(',')[1][:-1]
            
            urls = build_urls(endpoints, orgao_id, cpf, token)
            for i, url in enumerate(urls):
                t1 = time.time()
                response = requests.get(url)
                t2 = time.time()

                if response.status_code == 200:
                    print(t2-t1, url)
                    response_times[i].append(t2 - t1)
