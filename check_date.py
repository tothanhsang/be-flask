from datetime import datetime

# # current date and time
# now = datetime.now()

# timestamp = datetime.timestamp(now)
# print("timestamp =", str(timestamp)[0:8] + '0000000')

timestamp = float('1599588000')
dt_object = datetime.fromtimestamp(timestamp)
print("dt_object =", dt_object)
print("type(dt_object) =", type(dt_object))


start_time = datetime.now().strftime('%Y/%m/%d') + ' 1:0'
print("start time 1: ", start_time)
start_time = datetime.strptime(start_time, '%Y/%m/%d %H:%M')
print("start time 2: ", start_time)
timestamp = datetime.timestamp(start_time)
print("timestamp =", str(timestamp).split(".")[0] + "000")

times = datetime.now().strftime('%Y/%m/%d') + ' 1:0'
times = datetime.strptime(times, '%Y/%m/%d %H:%M')
timestamp = str(datetime.timestamp(times)).split(".")[0] + "000"