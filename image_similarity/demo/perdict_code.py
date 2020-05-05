# coding: utf-8
import simplejson as json
import subprocess
import shlex
cmd = '''curl -X POST http://127.0.0.1:8080/predictions/arcface -F "img1=@c.jpg" -F "img2=@c2.jpg"'''
args = shlex.split(cmd)
process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()


stdout_j = json.loads(stdout)
print('stdout',stdout)
print('\n\nstderr',stderr)
print('\n\nstdout_j','Distance: ', stdout_j['Distance'], type(stdout_j))
