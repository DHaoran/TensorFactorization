# -*- coding=utf-8 -*-
import time
import numpy as np
from ntflib import utils
from ntflib import betantf

TIME_SLOT_FIRST = 0
TIME_SLOT_SECOND = 1
TIME_SLOT_THIRD = 2

class TensorProcess():

	def __init__(self):
		self.speeds = {}
		self.file_name = "./speed_end.txt"

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
			if courier not in self.speeds:
				self.speeds[courier] = {}
			if edge_id not in self.speeds[courier]:
				self.speeds[courier][edge_id] = {}
			time_slot = self.getTimeSlot(timestamp)
			if time_slot not in self.speeds[courier][edge_id]:
				self.speeds[courier][edge_id][time_slot] = []
			self.speeds[courier][edge_id][time_slot].append((dist, during))
	
	# 生成速度的张量
	def generate_tensor(self):
		for courier_edge in self.merge_courier_edges:
			first_courier, second_courier, union_edges = courier_edge
			couriers = (first_courier, second_courier)
			time_slots = (TIME_SLOT_FIRST, TIME_SLOT_SECOND, TIME_SLOT_THIRD)
			#time_slots = (TIME_SLOT_FIRST,)
			edges = list(union_edges)
			speeds_tensor = np.zeros(shape = (len(couriers), len(edges), len(time_slots)), dtype = float)
			count = 0
			count_low_speed = 0
			for x in range(len(couriers)):
				courier = couriers[x]
				for y in range(len(edges)):
					edge_id = edges[y]
					for z in range(len(time_slots)):
						time_slot = time_slots[z]
						if edge_id in self.speeds[courier] and time_slot in self.speeds[courier][edge_id]:
							records = self.speeds[courier][edge_id][time_slot]
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
									speeds_tensor[x][y][z] = dist / during
									print speeds_tensor[x][y][z]
									if speeds_tensor[x][y][z] < 1:
										count_low_speed += 1
									count += 1
			print count_low_speed, count, len(couriers) * len(edges) * len(time_slots)
			self.tensor_factorization(speeds_tensor)
			return 0;
				
	# 进行张量分解
	def tensor_factorization(self, tensor):
		shape = tensor.shape
		rank = len(shape)
		x_indices = np.array([a.ravel() for a in np.indices(shape)]).T
		x_vals = tensor.ravel()
		bnf = betantf.BetaNTF(shape, n_components = 2, n_iters = 30)
		initial = bnf.impute(x_indices)
		print "impute"
		initial.shape = shape
		reconstructed = bnf.fit(x_indices, x_vals)
		x_vals.tofile("x_vals.txt")
		np.savetxt("/Users/weikui/File/source/pyFile/NTFLib/x_vals.txt", x_vals)
		reconstructed.shape = shape
		#print reconstructed[0]
		np.savetxt("/Users/weikui/File/source/pyFile/NTFLib/reconstructed.txt", reconstructed.ravel())
		#self.output(reconstructed)

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

	# 获取时间戳所在的时间槽
	def getTimeSlot(self, timestamp):
		#return TIME_SLOT_FIRST
		timestamp = int(timestamp[:10])
		ltime = time.localtime(timestamp)
		if ltime.tm_hour >= 8 and ltime.tm_hour < 10:
			return TIME_SLOT_FIRST
		elif ltime.tm_hour >= 16 and ltime.tm_hour < 19:
			return TIME_SLOT_SECOND
		else:
			return TIME_SLOT_THIRD

	# 获取考虑的两个快递员编号和相应的边
	def getEdgesForCourier(self):
		sortedCouriers = sorted(self.speeds.items(), key = lambda d:len(d[1]), reverse = True)
		couriers_edges = []
		for courierRecord in sortedCouriers:
			courier_edge = {}
			courier_edge['id'] = courierRecord[0]
			courier_edge['edges'] = set()
			for edge in courierRecord[1]:
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
		self.merge_courier_edges = []
		for sortedCourier in sortedMerges:
			first_courier = sortedCourier[0][0]
			second_courier = sortedCourier[0][1]
			union_edges = sortedCourier[1]['union']
			#print first_courier, second_courier, len(union_edges), len(sortedCourier[1]['merge']), union_edges
			#merge_courier_edges[(first_courier, second_courier)] = union_edges
			self.merge_courier_edges.append((first_courier,second_courier,union_edges))
		#return merge_courier_edges
	
if __name__ == '__main__':
	tensorProcess = TensorProcess()
	tensorProcess.process()
	tensorProcess.getEdgesForCourier()
	tensorProcess.generate_tensor()
