# -*- coding=utf-8 -*-
import time
import numpy as np
from ntflib import utils
from ntflib import betantf

SPLIT_X = 3
SPLIT_Y = 10
SPLIT_Z = 10

class TensorProcess():
	TIME_SLOT_FIRST = 0
	TIME_SLOT_SECOND = 1
	TIME_SLOT_THIRD = 2

	def __init__(self):
		self.couriers = {}
		self.edges = {}
		self.index_to_courier = {}
		self.index_to_edge = {}
		self.file_name = "./speed_end.txt"
		self.X = 3
		self.Y = 0
		self.Z = 0

	# 处理速度文件
	def process(self):
		for index,line in enumerate(open(self.file_name).readlines()):
			if index == 0:
				continue
			records = line.strip().split(',')
			edge_id = records[1]
			dist = float(records[2])
			during = int(records[3])
			courier = records[4]
			timestamp = records[5]
			if courier not in self.couriers:
				courier_num = len(self.couriers)
				self.couriers[courier] = {}
				self.couriers[courier]['id'] = courier_num
				self.index_to_courier[courier_num] = courier
				self.couriers[courier]['record'] = {}
			if edge_id not in self.couriers[courier]['record']:
				self.couriers[courier]['record'][edge_id] = {}
			time_slot = self.getTimeSlot(timestamp)
			if time_slot not in self.couriers[courier]['record'][edge_id]:
				self.couriers[courier]['record'][edge_id][time_slot] = []
			self.couriers[courier]['record'][edge_id][time_slot].append((dist, during))
			if edge_id not in self.edges:
				edge_num = len(self.edges)
				self.edges[edge_id] = edge_num
				self.index_to_edge[edge_num] = edge_id
		self.Y = len(self.couriers)
		self.Z = len(self.edges)
	
	# 获取时间戳所在的时间槽
	def getTimeSlot(self, timestamp):
		timestamp = int(timestamp[:10])
		ltime = time.localtime(timestamp)
		if ltime.tm_hour >= 8 and ltime.tm_hour < 10:
			return self.TIME_SLOT_FIRST
		elif ltime.tm_hour >= 16 and ltime.tm_hour < 19:
			return self.TIME_SLOT_SECOND
		else:
			return self.TIME_SLOT_THIRD

	# 生成张量的矩阵
	def generate_tensor(self, k = 2):
		self.speeds_tensor = np.zeros(shape = (self.X, self.Y, self.Z), dtype = float)
		count = 0
		count_low_speed = 0
		for courier in self.couriers:
			y = self.couriers[courier]['id']
			for edge_id in self.couriers[courier]['record']:
				z = self.edges[edge_id]
				for time_slot in self.couriers[courier]['record'][edge_id]:
					x = time_slot
					records = self.couriers[courier]['record'][edge_id][time_slot]
					dist = 0
					during = 0
					for record in records:
						if not record[1] > 10:
							dist += record[0]
							during += record[1]
					if (during == 0) :
						pass
					else:
						speed = dist / during
						if speed < 0.5 or speed > 12:
							pass
						else:
							self.speeds_tensor[x][y][z] = dist / during
							#if x == self.TIME_SLOT_SECOND:
							print self.speeds_tensor[x][y][z]
							if self.speeds_tensor[x][y][z] < 1:
								count_low_speed += 1
							count += 1
		print count, count_low_speed, self.X * self.Y * self.Z, x * 1.0 / (self.X * self.Y * self.Z), count_low_speed * 1.0 / count

		#self.change_tensor()
		shape = self.speeds_tensor.shape
		rank = len(shape)
		x_indices = np.array([a.ravel() for a in np.indices(shape)]).T
		x_vals = self.speeds_tensor.ravel()
		return shape, rank, k, x_indices, x_vals
		#print self.speeds_tensor
		"""
		shape = (len(self.couriers), len(self.edges), 3)
		rank = len(shape)
		init = [gen_rand(s, k) for s in shape]
		hidden = [gen_rand(s, k) for s in shape]
		x = parafac(hidden)
		x_indices = np.array([a.ravel() for a in np.indices(shape)]).T
		x_vals = x.ravel()
		return shape, rank, k, init, x, x_indices, x_vals
		"""
	
	# 为了处理方便,进行张量的截取
	def change_tensor(self):
		tmp = np.zeros(shape = (SPLIT_X, SPLIT_Y, SPLIT_Z), dtype = float)
		for x in range(SPLIT_X):
			for y in range(SPLIT_Y):
				for z in range(SPLIT_Z):
					tmp[x][y][z] = self.speeds_tensor[x][y][z]
		self.speeds_tensor = tmp
	
	# 进行张量分解
	def tensor_factorization(self):
		shape, rank, k, x_indices, x_vals = self.generate_tensor()
		bnf = betantf.BetaNTF(shape, n_components = k, n_iters = 4)
		initial = bnf.impute(x_indices)
		print "impute"
		initial.shape = shape
		self.reconstructed = bnf.fit(x_indices, x_vals)
		x_vals.tofile("x_vals.txt")
		np.savetxt("/Users/weikui/File/source/pyFile/NTFLib/x_vals.txt", x_vals)
		self.reconstructed.shape = shape
		#print reconstructed[0]
		np.savetxt("/Users/weikui/File/source/pyFile/NTFLib/reconstructed.txt", self.reconstructed.ravel())

	# 将张量输出
	def output(self, tensor):
		shape = tensor.shape
		writer = open("speed_dense_end.txt", "w")
		writer.write("快递员编号, 道路编号, 时间槽, 速度\n")
		for x in range(shape[0]):
			for y in range(shape[1]):
				courier = self.index_to_courier[y]
				for z in range(shape[2]):
					edge_id = self.index_to_edge[z]
					writer.write("%s,%s,%s,%s\n" % (courier, edge_id, x, tensor[x][y][z]))
		writer.close()

	def getEdgesForCourier(self):
		sortedCouriers = sorted(self.couriers.items(), key = lambda d:len(d[1]['record']), reverse = True)
		couriers_edges = []
		for courier in sortedCouriers:
			#print courier[0], courier[1]
			courier_edge = {}
			courier_edge['id'] = courier[0]
			courier_edge['edges'] = set()
			for edge in courier[1]['record']:
				courier_edge['edges'].add(edge)
			couriers_edges.append(courier_edge)
		merge_couriers = {}
		for i in range(len(couriers_edges) - 1):
			for j in range(i + 1, len(couriers_edges)): 
				first = couriers_edges[i]
				second = couriers_edges[j]
				merge_couriers[(first['id'], second['id'])] = {}
				merge_couriers[(first['id'], second['id'])]['merge'] = first['edges'] & second['edges'] 
				merge_couriers[(first['id'], second['id'])]['union'] = first['edges'] | second['edges'] 
		sortedMerges = sorted(merge_couriers.items(), key = lambda d: len(d[1]['merge']), reverse = True)
		for sortedCourier in sortedMerges:
			print sortedCourier[0][0], sortedCourier[0][1], len(sortedCourier[1]['merge']), len(sortedCourier[1]['union']),sortedCourier[1]['merge'], sortedCourier[1]['union']
		#for courier in self.couriers:
		#	print courier, len(self.couriers[courier]['record'])
	
if __name__ == '__main__':
	tensorProcess = TensorProcess()
	tensorProcess.process()
	tensorProcess.getEdgesForCourier()
	#tensorProcess.generate_tensor()
	#tensorProcess.output(tensorProcess.speeds_tensor)
	#tensorProcess.tensor_factorization()
	#tensorProcess.output(tensorProcess.reconstructed)
	print len(tensorProcess.couriers)
	print len(tensorProcess.edges)
