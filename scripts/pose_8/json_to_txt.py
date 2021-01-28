'''
code by zzg@2021/01/16
function: convert json to txt
'''
import os
import json

json_file = "/workspace/zigangzhao/0116_Card/data_0104/image_and_annatation/Driver_front.json"
txt_path = json_file[:-5] + '_' + 'txt' 

if not os.path.exists(txt_path):
    os.makedirs(txt_path)

f = open(json_file)
setting = json.loads(f.read())
# print(setting)
print(len(setting))
for x in setting:
    # print(setting[x])
    m = setting[x]
    file_name = m['filename']
    # print(file_name[:-3])
    region = m['regions']
    res1 = ""
    #res2 = ""
    cnt = 0
    for i in range(len(region)):
        #4 points or 8 points 
       cnt += 1
       x1 = region[i]['shape_attributes']['cx']
       y1 = region[i]['shape_attributes']['cy']
       res1 += " " + str(x1) + " " + str(y1)
       res1 = res1.strip()
    print(res1)
    print(cnt)
    if cnt < 8:
       res1 = res1 + ' ' + res1
     
    f1 = open(txt_path + '/' + file_name.replace('jpg', 'txt'), 'w')
    if len(res1) >= 1:
        f1.write(res1 + '\n')
        f1.close()


