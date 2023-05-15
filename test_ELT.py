import os
import shutil
import datetime as dt
import csv
import json

PERIOD_LENGTH = 5  # minutes
SOURCE_DIR = os.path.join('.', 'tom-tom-data')
TRAIN_DIR = os.path.join('.', 'is_hot_tomtom_segment_status2')

if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)

def velocity_to_los(velocity):
    if velocity < 15:
        return 'F'
    elif velocity < 20:
        return 'E'
    elif velocity < 25:
        return 'D'
    elif velocity < 30:
        return 'C'
    elif velocity < 35:
        return 'B'
    else:
        return 'A'


def los_to_velocity(los):
    los_to_velocity = {
        'A': 35,
        'B': 30,
        'C': 25,
        'D': 20,
        'E': 15,
        'F': 10,
    }
    return los_to_velocity[los] or 45


def parse_date_and_period(timestamp):
    ts = dt.datetime.fromtimestamp(timestamp)
    date, time, weeekday = ts.date(), ts.time(), ts.weekday()

    h, m, s = time.hour, time.minute, time.second

    hour = f"0{h}" if h < 10 else str(h)
    step = (m * 60 + s) // (PERIOD_LENGTH * 60)
    m = PERIOD_LENGTH * step
    minute = f"0{m}" if m < 10 else str(m)
    period = f"period_{hour}_{minute}"

    return str(date), period, weeekday


def reset():
    shutil.rmtree(TRAIN_DIR)

def get_period_from_timestamp(timestamp):
	timestamp = dt.datetime.fromtimestamp(timestamp)
	hour = timestamp.hour
	minute = timestamp.minute

	if (hour >= 0 and hour <= 5) or (hour >= 9 and hour <= 15) or (hour >= 19 and hour <= 23):
		return "period_{hour}".format(hour=hour)
	if (hour == 24):
		return 'period_0'
	if (minute >= 30):
		return "period_{hour}_30".format(hour=hour)
	return "period_{hour}".format(hour=hour)

def get_seg_weather_data(timestamp, weather_data):
    weather = ""
    temperature = ""
    try:
        time_data = weather_data[str(timestamp)]
        weather = time_data["weather"][0]["main"]
        temperature = time_data["main"]["temp"]
    except:
        pass
    return weather, temperature

def main_is_hot_without_weather_data():
	index = dict()
	# base_data = json.load(open('base_status.json', 'r'))
	output2 = []
	index_2 = 0
	file_path2= os.path.join(TRAIN_DIR, "total_results.csv")
	for f in os.listdir(SOURCE_DIR):
		try:
			timestamp = f.split('.')[0]
			date, period, weekday = parse_date_and_period(int(timestamp))

			filename = period + '.csv'
			folder_path = os.path.join(TRAIN_DIR, date)
			file_path = os.path.join(TRAIN_DIR, date, filename)

			# Prepare
			if not os.path.exists(folder_path):
				os.makedirs(folder_path)

			header = ['id', 'period', 'segment_id',
					'date', 'weekday', 'tomtom_velocity', 'base_LOS', 'isHot', 'weather', 'temperature']
			if not os.path.exists(file_path):
				index[f"{date}/{period}"] = 0
				with open(file_path, newline='', mode='w') as new_file:
					writer = csv.writer(new_file)
					writer.writerow(header)

			# Read data
			output = []
			with open(os.path.join(SOURCE_DIR, f), 'r') as in_file:
				data = json.load(in_file)
				for k, v in data.items():
					isHot = True
					if v['source'] != 'tom-tom' or not isHot: continue
					base_LOS = velocity_to_los(v['velocity'])

					output.append([index[f"{date}/{period}"], period, k, date, 
						weekday, v['velocity'], base_LOS, isHot, "few clouds", 27])
					output2.append([index_2, period, k, date, 
						weekday, v['velocity'], base_LOS, isHot, "few clouds", 27])
					index_2 += 1
					print(f, output[-1])
					index[f"{date}/{period}"] += 1
			# Write transformed data
			with open(file_path, newline='', mode='a+') as out_file:
				writer = csv.writer(out_file)
				writer.writerows(output)
		except:
			continue
	header = ['id', 'period', 'segment_id',
					'date', 'weekday', 'tomtom_velocity', 'base_LOS', 'isHot', 'weather', 'temperature']
	with open(file_path2, newline='', mode='a+') as out_file2:
		writer = csv.writer(out_file2)
		writer.writerow(header)
		writer.writerows(output2)
		
reset()
main_is_hot_without_weather_data()