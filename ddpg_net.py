
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import math
from numpy import *

DELTA = 0.1
EPSILON = 1e-2 #MB
INT_MAX = double(2147483647)
BANDWIDTH_FACTOR = double(1.)
BANDWIDTH_NORMALIZE_FACTOR = double(5120.0)
BASELINE = False
MAXI_P = 4
PRINT_FLAG = False

class Flow(object):

	def __init__(self, arrival = 0, src = 1, dst = 2, size = 10, deadline = -1, flow_id = 0):
		self.src = src
		self.dst = dst
		self.size = double(size)
		self.original_size = double(size)
		self.ddl = deadline
		self.start = arrival
		self.flow_id = int(flow_id)
		self.rate = 0.
		self.que_id = 0.
		self.rate_vec = [0.] * MAXI_P
		self.path_rate_vec = [INT_MAX] * MAXI_P

class NetEnv(object):

	def __init__(self, gml, tracefile, visible = 25, maxi_path = 4):

		self.net_nodes = 0.
		self.graph = nx.DiGraph()
		self.allo_graph = nx.DiGraph()
		self.asap = {}
		self.sp = {}
		self.maxi_path = maxi_path
		MAXI_P = maxi_path
		self.visible = visible

		self.scheduler = []
		self.TRACE = []
		self.running_flow = []
		self.sub_scribe_links = {}

		self.CURRENT_TIME = double(0)
		self.LAST_TIME = double(0)

		self.TOTAL_FCT = double(0.)
		self.FINISHED = 0
		self.COUNT = 0

		self.scheduler_len = 0.
		self.DATA_MEAN = 0.
		self.DATA_STD = 0.

		self.last_remaining = 0.
		self.TOTAL_FLOWS = 0
		self.per_time_volume = 0.

		with open(gml, 'r') as topology:
			lines = topology.readlines()
			# number of nodes
			self.net_nodes = int (lines[0])
			for i in xrange(1, self.net_nodes + 1):
				self.graph.add_node(i)
				self.allo_graph.add_node(i)
			
			# the first line is number of nodes
			for line in lines[1:]:
				src, dst, bw = line.split()
				self.graph.add_edge(int(src), int(dst), weight = double(bw) * BANDWIDTH_FACTOR)
				self.graph.add_edge(int(dst), int(src), weight = double(bw) * BANDWIDTH_FACTOR)
			
			# tenant layer -- a full mesh
			for i in xrange(1, self.net_nodes + 1):
				for k in xrange(1, self.net_nodes + 1):
					self.allo_graph.add_edge(i, k, weight = 0.)

			self.co_graph = self.allo_graph.copy()
			self.residual_graph = self.allo_graph.copy()

			for i in xrange(1, self.net_nodes + 1):
				for k in xrange(1, self.net_nodes + 1):
					if i != k:
						index = str(i) + '_' + str(k)
						path_iter = nx.all_simple_paths(self.graph, i, k)

						paths = []
						for p in path_iter:
							paths.append(p)

						self.asap[index] = paths

			self.refine_asap()
			self.load_data_trace(tracefile)

	def get_path_bw(self, p, graph):
	    bw = INT_MAX

	    for i in xrange(len(p) - 1):
	        bw = min(bw, graph[p[i]][p[i + 1]]['weight'])

	    return bw

	def refine_asap(self):
		for key in self.asap:
			value = self.asap[key]
			for ii in xrange(len(value) - 1):
				for kk in xrange(len(value) - ii - 1):
					if len(value[kk]) > len(value[kk + 1]):
						value[kk], value[kk + 1] = value[kk + 1], value[kk]

			self.asap[key] = value[:self.maxi_path]
			self.sp[key] = value[0]

	def load_data_trace(self, file):
		last_start = -1
		data_size = []
		id_ = 0
		flow_id = 0

		with open(file, 'r') as trace:
			lines = trace.readlines()

			for line in lines:
				line = (line.strip()).split()
				[start, src, dst, size, ddl] = [double(line[0]), int(line[1]), int (line[2]), \
												double(line[3]), double(line[4])]

				assert(start >= last_start)

				nflow = Flow(start, src, dst, size, ddl, flow_id)
				flow_id += 1
				self.TRACE.append(nflow)
				data_size.append(size)
				last_start = start

		self.TOTAL_FLOWS = flow_id
		# normalize the size
		array_ = np.array(data_size)
		self.DATA_MEAN = array_.mean()
		self.DATA_STD = array_.std()

		self.LAST_TIME = self.CURRENT_TIME = double(self.TRACE[0].start)

		while len(self.TRACE) > 0 and self.TRACE[0].start <= self.CURRENT_TIME:
			self.scheduler.append(self.TRACE[0])
			del self.TRACE[0]

		assert(len(self.scheduler) > 0)
		self.scheduler_len = len(self.scheduler)
		self.sort_scheduler()

	def refresh_graph(self, path, allocated_bw):
		for i in xrange(len(path) - 1):
			self.graph[path[i]][path[i + 1]]['weight'] -= allocated_bw

			if self.graph[path[i]][path[i + 1]]['weight'] < 0:
				print '#: ', self.graph[path[i]][path[i + 1]]['weight']
				assert (self.graph[path[i]][path[i + 1]]['weight'] >= 0)

	def get_obser(self):
		obser = [0.] * (self.net_nodes ** 2)
		for ii in xrange(self.net_nodes):
			for kk in xrange(self.net_nodes):
				obser[ii * self.net_nodes + kk] = self.allo_graph[ii + 1][kk + 1]['weight']/BANDWIDTH_NORMALIZE_FACTOR
		
		src_que = [0.] * self.visible
		dst_que = [0.] * self.visible
		size_que = [0.] * self.visible
		ddl_que = [0.] * self.visible

		flow_info = [0.] * (3 * self.visible)

		# state space normalization
		cur_visible = min(len(self.scheduler), self.visible)
		for i, item in enumerate(self.scheduler[:cur_visible]):
			flow_info[i * 3] = item.src/double(self.net_nodes)
			flow_info[i * 3 + 1] = item.dst/double(self.net_nodes)
			flow_info[i * 3 + 2] = (item.size - self.DATA_MEAN)/(self.DATA_STD + 1e-6)

			'''
			#src_que[i] = item.src/double(self.net_nodes)
			#dst_que[i] = item.dst/double(self.net_nodes)
			#size_que[i] = (item.size - self.DATA_MEAN)/(self.DATA_STD + 1e-6)
			'''

		#print 'obser....', obser
		obser = obser + flow_info#src_que + dst_que + size_que

		#print 'observation....', obser

		return array(obser)

	def get_time(self):
		return self.CURRENT_TIME

	def max_min_fairness(self, rate_vec, capacity):
		cur_len = 0.
		allocated_bw = 0.
		result = [0.] * len(rate_vec)
		original_ = capacity

		for item in rate_vec:
			if item > 0:
				cur_len += 1

		'''
		if cur_len > 0:
			result = [min(double(x)/sum(rate_vec) * capacity, x) for x in rate_vec]
			allocated_bw = capacity
		
		'''
		while cur_len > 0 and capacity > 1e-5:
			ratio = capacity / double(cur_len)

			for i, item in enumerate(rate_vec):
				if item > 0:
					allocated = min(ratio, item)

					rate_vec[i] -= allocated
					result[i] += allocated
					allocated_bw += allocated
					capacity -= allocated

					if rate_vec[i] == 0:
						cur_len -= 1
		
		#print 'point 2...'
		assert (original_ - allocated_bw > -0.0001)
		
		return result, allocated_bw

	def subscribe_links(self, flow_id):
		actual_rate_vec = [0.] * self.maxi_path

		for key in self.sub_scribe_links:
			rate_vec = [(item[0].rate_vec[item[1]]) * item[2] for item in self.sub_scribe_links[key]]
			keys = key.split('_')
			src, dst = [int(keys[0]), int(keys[1])]

			#print 'previous,,,', rate_vec
			real_rate_vec, allocated_bw = self.max_min_fairness(rate_vec, self.graph[src][dst]['weight'])
			self.allo_graph[src][dst]['weight'] += sum(rate_vec)
			self.residual_graph[src][dst]['weight'] = self.graph[src][dst]['weight'] - allocated_bw

			for id_, item in enumerate(self.sub_scribe_links[key]):		# decide the rate of path
				current_bw = item[0].path_rate_vec[item[1]]
				item[0].path_rate_vec[item[1]] = min(current_bw, real_rate_vec[id_]) # should some residual bandwidth
				if item[0].flow_id == flow_id:
					actual_rate_vec = item[0].path_rate_vec

		return actual_rate_vec

	def back_to_running(self):
		Hash_table = [0] * (self.TOTAL_FLOWS + 5) 

		for key in self.sub_scribe_links:
			for item in self.sub_scribe_links[key]:
				flow = item[0]

				if Hash_table[flow.flow_id] == 0:
					total_rate = double(0)
					for item in flow.path_rate_vec:
						if item != INT_MAX:
							total_rate += item
					flow.rate = total_rate
					self.running_flow.append(flow)

				Hash_table[flow.flow_id] = 1

		assert(self.scheduler_len == len(self.running_flow))

	def subscribe_path(self, paths, cur_flow):
		for ii, item in enumerate(cur_flow.rate_vec):
			path_bw = self.get_path_bw(paths[ii], self.graph)

			for kk in xrange(len(paths[ii]) - 1):
				link_id = str(paths[ii][kk]) + '_' + str(paths[ii][kk + 1])

				if link_id not in self.sub_scribe_links:
					self.sub_scribe_links[link_id] = []

				self.sub_scribe_links[link_id].append([cur_flow, ii, path_bw])

	def residual_bw_check(self):

		for flow in self.running_flow:
			i, k = flow.src, flow.dst
			paths = self.asap[str(i) + '_' + str(k)]
			residual_bw = [self.get_path_bw(x, self.residual_graph) for x in paths]
			print i, '_', k, ': ', residual_bw

	def step(self, action):

		cur_flow = self.scheduler[0]
		flow_id = cur_flow.flow_id
		reward = 0.
		done = False

		src_dst_id = str(cur_flow.src) + '_' + str(cur_flow.dst)

		paths = self.asap[src_dst_id]
		if BASELINE == True:
			action = [1.] * self.maxi_path 

		cur_flow.rate_vec = action
		cur_flow.path_rate_vec = [double(INT_MAX)] * self.maxi_path
		del self.scheduler[0]

		self.subscribe_path(paths, cur_flow)

		if len(self.scheduler) == 0 or BASELINE == False:
			self.allo_graph = self.co_graph.copy()
			actual_vec = self.subscribe_links(flow_id)

		if len(self.scheduler) == 0:
			self.back_to_running()

		# decision-series reward
		expect_total = 0.

		if BASELINE == False:
			total_rate = 0.
			#print 'action...', action
			for i, rate in enumerate(action):
				expect_rate = rate * self.get_path_bw(paths[i], self.graph)
				#reward += max((actual_vec[i] - expect_rate) * 0.1, -50)
				total_rate += actual_vec[i]
				expect_total += expect_rate
			
			'''if total_rate > 0:
				finish_t = cur_flow.size/total_rate
				if finish_t < 0.5:
					reward += (0.5 - finish_t) * 1500
			'''
		obser = self.get_obser()

		if len(self.scheduler) == 0:
			self.find_time()
			reward = self.get_final_reward() # FCT
			done = self.walk_through()
			obser = self.get_obser()
			
		return obser, reward, done

	def get_scheduler_len(self):
		return len(self.scheduler)

	def get_final_reward(self):
		return -self.last_remaining * self.scheduler_len


	def find_time_(self):

		remaining_time = double(INT_MAX)

		for flow in self.running_flow:
			if flow.rate > 1e-3:
				remaining_time = min(remaining_time, flow.size/double(flow.rate))

		if remaining_time > 10:
			remaining_time = 100

		elif remaining_time > 0.5:
			remaining_time = 0.5

		return remaining_time * self.scheduler_len

	def find_time(self):

		remaining_time = double(INT_MAX)

		for flow in self.running_flow:
			if flow.rate > 1e-3:
				remaining_time = min(remaining_time, flow.size/double(flow.rate))

		self.LAST_TIME = self.CURRENT_TIME
		self.CURRENT_TIME = self.CURRENT_TIME + remaining_time

		if len(self.TRACE) > 0 and self.TRACE[0].start < self.CURRENT_TIME:
			self.CURRENT_TIME = self.TRACE[0].start
			assert (self.LAST_TIME <= self.TRACE[0].start)

		if self.CURRENT_TIME - self.LAST_TIME > 0.5:
			self.CURRENT_TIME = self.LAST_TIME + 0.5

		self.last_remaining = self.CURRENT_TIME - self.LAST_TIME

		if remaining_time > 10:
			self.last_remaining = 100

	def get_total_fct(self):
		return self.TOTAL_FCT

	def get_per_volume(self):
		return self.per_time_volume

	def sort_scheduler(self):
		for ii in xrange(len(self.scheduler) - 1):
			for jj in xrange(len(self.scheduler) - ii - 1):
				if self.scheduler[jj].size > self.scheduler[jj + 1].size:
					self.scheduler[jj], self.scheduler[jj + 1] = self.scheduler[jj + 1], self.scheduler[jj]

	def walk_through(self):
		self.per_time_volume = 0.
		self.sub_scribe_links = {}
		assert (len(self.scheduler) == 0)

		# walk through the past
		past_interval = self.CURRENT_TIME - self.LAST_TIME
		cur_running = len(self.running_flow)

		for id_, flow in enumerate(self.running_flow):
			self.per_time_volume += past_interval * double(flow.rate)
			remaining_size = flow.size - past_interval * double(flow.rate)

			if remaining_size < -1 or PRINT_FLAG == True:
				print remaining_size
				print 'original size:\t', flow.original_size
				print 'last size:\t', flow.size
				print 'flow rate:\t', flow.rate
				print 'remaining size:\t', remaining_size
				print 'start time:\t', flow.start
				print 'interval:\t', past_interval
				print 'last time:\t', self.LAST_TIME
				print 'current time:\t', self.CURRENT_TIME
				assert (remaining_size > -1)

			if remaining_size <= EPSILON:
				self.TOTAL_FCT += (self.CURRENT_TIME - flow.start)
				self.FINISHED += 1
			else:
				flow.size = remaining_size
				self.scheduler.append(flow)

		# schedule the running flow by the increasing order of FCT?
		while len(self.TRACE) > 0 and self.TRACE[0].start <= self.CURRENT_TIME:
			self.scheduler.append(self.TRACE[0])
			if self.TRACE[0].start < self.CURRENT_TIME:
				assert (4==0)
			del self.TRACE[0]

		self.sort_scheduler()

		self.COUNT += 1

		if self.COUNT % 50 == 0:
			print 'remaining jobs: ', len(self.TRACE),'\t', 'scheduling jobs: ', len(self.scheduler), '\n finished jobs: ',\
			  self.FINISHED, '\t Total jobs: ', (len(self.TRACE) + len(self.scheduler) + self.FINISHED), '\n'

		self.running_flow = []
		self.scheduler_len = len(self.scheduler)

		return (self.scheduler_len == 0 and len(self.TRACE) == 0)

DATA_TRACE = 'flows_swan.txt'
env = NetEnv('swan.txt', DATA_TRACE)
#env.load_data_trace()
