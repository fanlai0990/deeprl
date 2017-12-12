import os

def generator(file, new_file):
	with open(file, 'r') as input:
		lines = input.readlines()

		# skip the first several lines
		for i, line in enumerate(lines):
			line = line.strip().split()
			#print line
			if line[0] == 'node':
				lines = lines[i:]
				break
		#print lines
		NumofNodes = -1

		while True:
			line = lines[NumofNodes * 4 + 4].strip().split()
			if line[0] == 'node':
				NumofNodes += 1
			else:
				lines = lines[NumofNodes * 4 + 4:]
				break

		src = []
		dst = []
		bw = []
		print lines
		for i in xrange(len(lines)):
			src.append(lines[i*5+1].strip().split()[1])
			dst.append(lines[i*5+2].strip().split()[1])
			print lines[i*5+3].strip().split()
			bw.append(lines[i*5+3].strip().split()[1])
			if (i+1)*5 + 3 > len(lines):
				break

		output = open(new_file, 'w')
		#with open(new_file, 'w') as output:
		output.writelines(str(NumofNodes) + '\n')
		for i in xrange(len(src)):
			output.writelines(src[i] + '\t' + dst[i] + '\t' + bw[i] + '\n')

		print 'finish...'


'''path = "/Users/laifan/documents/github/gaia/src/main/java/gaiasim/scheduler/drlscheduler/new_drl/gml" 
files= os.listdir(path) 
print files
s = []  
for file in files:
     if not os.path.isdir(file):
          f = path+"/"+file
          if '.gml' in f:
         	 generator(f, f.replace('gml','txt'))
'''
generator('./gml/gb4.gml', './gml/gb4.txt')
print 'finish all'