#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
import requests
import time
from pytz import timezone
import datetime as dt
import schedule




API_KEYS = [
  '21HYqiZMMrc5SrNN6jVOk230BWbg3tAj',
  'Pm7CnuOA6G6Uw25xaMtGy6yxSmaqACXM',
  'cBAeCxqGO4OJq7y1wfyFOhA0HRWpgAml',
  'nGKarNyeU0VtPjtM5chG4Uif6KlRUa1E',
  '1j5sGYeflskicFlJYJtMbgFdLB9xRAji'
]

api_counters = [0] * len(API_KEYS)
api_call_counter = 0
retry = 0
MAX_RETRY = 10
LOG_FILE = 'tomtom_logs.txt'

checked_time_list = {
   "1": ["15:57:26", "18:57:26"],
   "2": ["6:28:00", "10:32:00"]
}

STYLES = ['absolute', 'relative', 'relative0', 'relate0-dark', 'relative-delay', 'reduced-sensitivity']
ZOOM = 22
TOMTOM_URL = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/{STYLES[0]}/{ZOOM}/json"


API_KEYS_WEATHER = [
  '8df3c71da563b1c6a06e2239c780ba82'
]

LOG_FILE = 'weatherapi_logs.txt'
HCMC = [10.8333, 106.6667]	# https://openweathermap.org/find?q=Ho+Chi+Minh

BASE_URL = f"https://api.openweathermap.org/data/2.5/weather"

def crawl_current_weather():
	params = {
		'lat' : HCMC[0],
		'lon' : HCMC[1],
		'appid' : API_KEYS_WEATHER[0]
	}
	data = requests.get(BASE_URL, params).json()
	result = "none"
	try:
		result = data["weather"][0]["main"]
	except:
		pass
	return result

def convert_str_to_time(time_str):
   return dt.datetime.strptime(time_str, '%H:%M:%S').time()

def check_times(list_time):
   tz_VN = timezone('Asia/Ho_Chi_Minh') 
   datetime_VN = dt.datetime.now(tz_VN)
   current_time = datetime_VN.time()
   checked_time = False
   for key, value in checked_time_list.items():
      if convert_str_to_time(value[0]) <= current_time <= convert_str_to_time(value[1]):
        checked_time = True

   return checked_time

def log(message):
  with open(LOG_FILE, 'a+') as f:
    f.write(message)
    

def tom_url(zoom_level): return f"https://api.tomtom.com/traffic/services/4/flowSegmentData/{STYLES[0]}/{zoom_level}/json"

def get_tomtom_data(lat, lng, zoom_level=22):
    global api_counters
    global api_call_counter

    try:
        params = {
          'point': f"{lat},{lng}",
          'unit': 'KMPH',
          'openLr': 'false',
          'key': API_KEYS[api_call_counter]
        }
        data = requests.get(tom_url(zoom_level), params=params).json()

        api_counters[api_call_counter] += 1
        return data['flowSegmentData']
        
    except ValueError:
        api_call_counter += 1
        return get_tomtom_data(lat, lng)
        
    except KeyError:
        log("Key error: " + str(lat) + "," + str(lng) + "\n")
        return None


# In[6]:


def crawl_data_with_location(lat, lng):
    """
    Main function to crawl data an
    d map to segment_id
    """
#     global checked_time_list
#     if not check_times(checked_time_list):
#        return

    tomtom_data = get_tomtom_data(lat, lng)
    return tomtom_data['currentSpeed']


# In[7]:


print(crawl_data_with_location(10.792465, 106.752099))


# In[ ]:




