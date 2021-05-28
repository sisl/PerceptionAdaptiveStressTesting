import colorsys
import numpy as np
import matplotlib.pyplot as plt
"""
extracts the Latitude and Longitude from oxts data file
for KITTI dataset
"""
def extractLatLon(file):
	with open(file) as r:
		data = r.readline()
		data = data.split(" ")
		return data[0], data[1]

"""
class Detected

Encapsulates a bounding box, predicted category, color (for visulation)
and index (for tracking unique objects).
"""
class Detected():
	def __init__(self, box, category, color, index):
		self.location = box[:3]
		self.boxDims = box[:]
		self.category = category
		self.color = color
		self.index = index
		self.timeStep = None
	def sameCatAs(self, other):
		return self.category == other.category

	def isDuplicatedObject(self, other):
		if self.category != other.category:
			return False
		#car
		if self.category == 1:
			x,y,z = self.location
			x2,y2,z2 = other.location
			if abs(x-x2) <= .7 and abs(y-y2) <= .7:
				return True
		#pedestrian
		if self.category == 2:
			x,y,z = self.location
			x2,y2,z2 = other.location
			if abs(x-x2) <= .3 and abs(y-y2) <= .3:
				return True
		#bicycle
		if self.category == 3:
			x,y,z = self.location
			x2,y2,z2 = other.location
			if abs(x-x2) <= .3 and abs(y-y2) <= .3:
				return True
		return False

	"""
	Sees if the Dectected (param) other could be the same object as (param) self
	given the (param) globalDistance. The constants used are just come from 
	how fast a car moves in a residential area about 15km/hr (these hyperparamters can
	be tuned) 
	"""
	def isPossibleChange(self, other, globalDistance):
		if self.category != other.category:
			return False
		#car
		if self.category == 1:
			if globalDistance <= 2:
				return True
		#pedestrian
		if self.category == 2:
			if globalDistance <= 1.2:
				return True
		#bicycle
		if self.category == 3:
			if globalDistance <= 1:
				return True
		return False
	def distanceTo(self, other):
		return np.linalg.norm(other.location - self.location)

"""
class Tracker

Keeps a list of all detected objects and, ego vehicle gps information.
Then for each new object added to the tracker, checks if that object was seen
in the previous timestep. If the new object is the same class and could feasibly have 
moved to the new spot in one timestep they are assumed to be the same object.
"""
class Tracker():
	def __init__(self):
		self.options = 12
		self.colors = [np.array(colorsys.hsv_to_rgb(i/self.options, 1, 1))*255 for i in range(self.options)]
		self.colorIndex = 0
		self.currentTime = -1
		self.histories = []
		self.pastCars = []
		self.pastPeds = []
		self.pastBikes = []
		self.trackedGPS = None
		self.scale = None
		self.offsets = []
	
	"""
	function cleanHistory

	KITTI dataset have sensors that update at 10Hz, but the nuscenes dataset updates at 2Hz
	so to align the datasets remove excess tracking data so that the KITTI data is also 2Hz.
	"""
	def cleanHistory(self):
		if self.currentTime % 5 != 0:
			return
		for i in range(len(self.histories)):
			newHistory = []
			for obj in self.histories[i]:
				if obj.timeStep % 5 == 0:
					newHistory.append(obj)
			self.histories[i] = newHistory

	def getHistory(self):
		return self.histories

	"""
	function getLocationInGlobal

	using the start gps data rotate the dectcted (param) obj location (which is in the local frame)
	to be in the global frame. Then add offest to the location based on how far the ego vehicle has moved 
	from its start position
	"""
	def getLocationInGlobal(self, obj):
		x,y,z = obj.location
		base_x, base_y, base_yaw = self.offsets[0]
		ego_x, ego_y, ego_yaw = self.offsets[-1]
		alignment = -ego_yaw

		translation_rotation = np.array([[np.cos(alignment), -np.sin(alignment)], [np.sin(alignment), np.cos(alignment)]])
		offset_x, offset_y, offset_yaw = ego_x-base_x, ego_y-base_y, ego_yaw-base_yaw
		offset_x, offset_y = translation_rotation @ np.array([offset_x, offset_y])

		rotation = np.array([[np.cos(offset_yaw), -np.sin(offset_yaw)], [np.sin(offset_yaw), np.cos(offset_yaw)]])
		x,y = rotation @ np.array([x,y])
		return x+offset_x, y+offset_y

	"""
	function getDistanceInGlobal

	calculate the L2 distance between obj1 and obj2 in the global frame
	"""
	def getDisctanceInGlobal(self, obj1, obj2):
		x1, y1 = self.getLocationInGlobal(obj1)
		x2, y2 = self.getLocationInGlobal(obj2)
		return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

	"""
	function addToHistory

	(param) current list of current objects
	(param) past    list of objects in the last timestep
	for each current object calculate the distance to the objects in the past 
	and if the closest past object is a feasible distance away from the current
	label the current object the same as the past object (give same color and index)
	otherwise given each object a new color and unique objects seen index.
	"""
	def addToHistory(self, past, current):
		for cur_object in current:
			distances = np.array([self.getDisctanceInGlobal(cur_object, past_object) for past_object in past])
			if len(distances) == 0:
				cur_object.index = len(self.histories)
				self.histories.append([cur_object])
				#give object color that is for testing
				cur_object.color = self.colors[self.colorIndex]
				self.colorIndex = (self.colorIndex + 1) % len(self.colors)
				continue
			closest_index = np.argsort(distances)[0]
			if cur_object.isPossibleChange(past[closest_index], distances[closest_index]):
				cur_object.index = past[closest_index].index
				self.histories[cur_object.index].append(cur_object)
				#give object color that is for testing
				cur_object.color = past[closest_index].color
			else:
				cur_object.index = len(self.histories)
				self.histories.append([cur_object])
				#give object color that is for testing
				cur_object.color = self.colors[self.colorIndex]
				self.colorIndex = (self.colorIndex + 1) % len(self.colors)
		self.cleanHistory()
	"""
	function assignColor

	(param) current list of current objects
	(param) past    list of objects in the last timestep
	This function is used only for testing that the objects are correctly detected and 
	shifted into the global frame
	for each current object calculate the distance to the objects in the past 
	and if the closest past object is a feasible distance away from the current
	label the current object the same as the past object (give same color and index)
	otherwise given each object a new color and unique objects seen index.
	"""
	def assignColor(self, past, current):
		for cur_object in current:
			distances = np.array([cur_object.distanceTo(past_object) for past_object in past])
			if len(distances) == 0: 
				cur_object.color = self.colors[self.colorIndex]
				self.colorIndex = (self.colorIndex + 1) % len(self.colors)
				continue
			index = np.argsort(distances)[0]
			if cur_object.isPossibleChange(past[index]):
				cur_object.color = past[index].color
			else:
				cur_object.color = self.colors[self.colorIndex]
				self.colorIndex = (self.colorIndex + 1) % len(self.colors)
	
	"""
	function calculateOffsets

	return the gps offsets from the current gps data
	"""
	def calculateOffsets(self):
		base_x, base_y, base_yaw = self.offsets[-1]
		return [(x-base_x, y-base_y, yaw-base_yaw) for x,y,yaw in self.offsets]

	def calculateOffsetsAtTime(self, time):
		base_x, base_y, base_yaw = self.offsets[time]
		return [(x-base_x, y-base_y, yaw-base_yaw) for x,y,yaw in self.offsets]

	"""
	function getOffsets

	return the gps offsets from the current gps data
	"""
	def getOffset(self, gps):
		if not self.trackedGPS:
			self.scale = np.cos(gps[0] * np.pi / 180)
		r = 6378137
		x = self.scale * r * np.pi * gps[1] / 180
		y = self.scale * r * np.log(np.tan(np.pi*(gps[0]+90) / 360))
		yaw = gps[2]
		if not self.trackedGPS:
			self.trackedGPS = (x, y, yaw)
			self.offsets.append((x, y, yaw))
			return self.calculateOffsets()
		self.offsets.append((x, y, yaw))
		return self.calculateOffsets()
	"""
	returns ths data structure needed to create an input to CoverNet
	"""
	def getTrackingInfo(self):
		if self.currentTime % 5 != 0:
			return None
		result = {"heading": self.offsets[-1][2], "gps": self.calculateOffsets(), "history": self.getHistory()}
		return result
	"""
	returns ths data structure needed to create an input to CoverNet at a specific timestep
	"""
	def getTrackingInfoAtTime(self, time):
		if time % 5 != 0:
			return None
		result = {"heading":self.offsets[time][2], "gps": self.calculateOffsetsAtTime(time), "history":self.getHistory()}
		return result
	
	"""
	function removeDuplicates
	(param) allObjects list of Detected objects
	returns the list of unique Detected object 
	"""
	def removeDuplicates(self, allObjects):
		isDuplicate = [False for _ in range(len(allObjects))]
		uniqueObjects = []
		for i in range(len(allObjects)):
			for j in range(i+1, len(allObjects)):
				if allObjects[i].isDuplicatedObject(allObjects[j]):
					isDuplicate[i] = True
			if not isDuplicate[i]:
				uniqueObjects.append(allObjects[i])
		return uniqueObjects
	
	"""
	function createDectedObjects
	(param) pred_dict dictionary produced by object detection (PV-RCNN) containing boundind box data,
	and prediction labels

	Iterates through pred_dict creating a Dectected objecr, removes duplicate objects
	and returns a tuple of the detected objects by predicted category (cars, pedestrian, bike, allobjects)
	"""
	def createDetectedObjects(self, pred_dict):
		self.currentTime += 1
		cars = []
		peds = []
		bikes = []
		allObjects = []
		for i,box in enumerate(pred_dict['pred_boxes']):
			allObjects.append(Detected(box.cpu().numpy(), pred_dict['pred_labels'][i], None, i))
		uniqueObjects = self.removeDuplicates(allObjects)
		
		for i in range(len(uniqueObjects)):
			uniqueObjects[i].timeStep = self.currentTime
			if uniqueObjects[i].category == 1:
				cars.append(uniqueObjects[i])
			if uniqueObjects[i].category == 2:
				peds.append(uniqueObjects[i])
			if uniqueObjects[i].category == 3:
				bikes.append(uniqueObjects[i])
		return cars, peds, bikes, uniqueObjects
	
	"""
	function updateHistory
	(param) pred_dict dictionary produced by object detection (PV-RCNN) containing boundind box data,
	and prediction labels
	
	Updates the tracking history given current prediction from an Object Detection model.
	"""
	def updateHistory(self, pred_dict):
		newCars, newPeds, newBikes, allObjects = self.createDetectedObjects(pred_dict)
		self.addToHistory(self.pastCars, newCars)
		self.addToHistory(self.pastPeds, newPeds)
		self.addToHistory(self.pastBikes, newBikes)
		self.pastCars = newCars
		self.pastPeds = newPeds
		self.pastBikes = newBikes
		return self.histories

	"""
	function getColorFunction
	(param) pred_dict dictionary produced by object detection (PV-RCNN) containing boundind box data,
	and prediction labels
	
	This function is used only for testing that Detected objects are being correctly displayed.
	"""
	def getColorFunction(self, pred_dict):
		newCars, newPeds, newBikes, allObjects = self.createDetectedObjects(pred_dict)
		self.assignColor(self.pastCars, newCars)
		self.assignColor(self.pastPeds, newPeds)
		self.assignColor(self.pastBikes, newBikes)
		self.pastCars = newCars
		self.pastPeds = newPeds
		self.pastBikes = newBikes
		return lambda i:allObjects[i].color

if __name__ == '__main__':
	pass
