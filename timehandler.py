from datetime import datetime
import time

def smallerThen(a,b):
	"""This function takes two string number and return if A is smaller then B"""
	if int(a) < int(b):
		return True
	else:
	 return False

def increaseOne(a):
	'''This function takes a string number and increase it by one and return as xx format'''
	return "{0:0>2}".format(str(int(a)+1))


class DateTimeHolder:
	'''This class simply just hold values of datetime. It holds day,month,year,hour,minute,second'''
	def __init__(self,day,month,year,hour,minute,second):
 		self.day = day
 		self.month = month
 		self.year = year
 		self.hour = hour
 		self.minute = minute
 		self.second = second

def getCurrentDateTime():
	'''On calling this returns current datetime as DateTimeHolder class.'''
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	dt_string = dt_string.split(' ')
	dt_string[0] = dt_string[0].split('/')
	dt_string[1] = dt_string[1].split(':')
	temp = dt_string[0] + dt_string[1]
	dt_string = temp

	# date month year hour minute second
	return DateTimeHolder(dt_string[0],dt_string[1],dt_string[2],dt_string[3],dt_string[4],dt_string[5])


class DateTimeHandler:
	'''A time handler holds current time, current frame capture time and feature extraction time.'''
	def __init__(self):
		self.temp = getCurrentDateTime()
		self.current_time = self.temp
		self.capture_time = DateTimeHolder(self.temp.day,self.temp.month,self.temp.year,self.temp.hour,self.temp.minute,self.temp.second)
		self.feature_time =  DateTimeHolder(self.temp.day,self.temp.month,self.temp.year,self.temp.hour,self.temp.minute,self.temp.second)
		self.daysinmonth = [31,28,31,30,31,30,31,31,30,31,30,31]


	def printDateTimeBeautifully(self,timeholder):
		"""This just simply prints a timeholder's data from a DateTimeHandler's timeholder. """
		print("Day: {}, Month: {}, Year: {}, Hour: {} , Minute: {}, Second: {}".format(timeholder.day,timeholder.month,timeholder.year,timeholder.hour,timeholder.minute,timeholder.second))
	
	def returnDateTimeString(self,timeholder):
		"""Returns a timeholder's data as string in {day_month_year_hour_munite_second} format."""
		return ("{}_{}_{}_{}_{}_{}".format(timeholder.day,timeholder.month,timeholder.year,timeholder.hour,timeholder.minute,timeholder.second))

	def getNextSecond(self,timeholder):
		'''Given a DateTimeHandler\'s timeholder, it updates the time to next second 
			
			Ex: dth.getNextSecond(dt.current_time)
		'''
		if smallerThen(timeholder.second , 60):
			timeholder.second = increaseOne(timeholder.second)
		else:
			timeholder.second = "01"
			if smallerThen(timeholder.minute, 60):
				timeholder.minute = increaseOne(timeholder.minute)
			else:
				timeholder.minute = "01"
				if smallerThen(timeholder.hour, 24):
					timeholder.hour = increaseOne(timeholder.hour)
				else:
					timeholder.hour = '01'
					if smallerThen(timeholder.day, self.daysinmonth[int(timeholder.month)-1]):
						timeholder.day = increaseOne(timeholder.day)
					else:
						timeholder.day = '01'
						if smallerThen(timeholder.month,12):
							timeholder.month = increaseOne(timeholder.month)
						else:
							timeholder.month = "1"
							timeholder.year = increaseOne(timeholder.year)

		


# dt = DateTimeHandler()
# # # print("date and time = ", dt.current_time.minute)
# dt.printDateTimeBeautifully(dt.current_time)

# # print(dt.returnDateTimeString(dt.current_time))

# # prev = dt.feature_time.year
# for _ in range(1,30):
# 	dt.getNextSecond(dt.feature_time)
# 	dt.printDateTimeBeautifully(dt.feature_time)
# time.sleep(2)
# dt.printDateTimeBeautifully(dt.current_time)