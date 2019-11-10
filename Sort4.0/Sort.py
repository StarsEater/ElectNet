# import json
#
# import pandas as pd
# import numpy as np
# import csv
# import re
# from fuzzywuzzy import fuzz
# import sys
#
# def is_json(myjson):
#     try:
#         json.loads(myjson)
#     except ValueError:
#         return False
#     return True
#
#
# def sort_reports(inputpath, json_str): #返回json串
#     word_dict = json.loads(json_str)
#     data = pd.read_csv(inputpath, encoding='utf-8')
#     data_list = []
#     for index, row in data.iterrows():
#         score = 100
#         # for key, value in word_list.items():
#         if word_dict['设备类型'] != 'null':
#             value = word_dict['设备类型'].replace('主变', '变压器')
#             if type(row['故障设备类型']) != float:
#                 string = str(row['故障设备类型']).replace('主变', '变压器')
#                 if value not in string:
#                     score = score - 20
#             else:
#                 score = score - 20
#         if word_dict['部件'] != 'null':
#             score = score - 20
#             value = word_dict['部件'].replace('主变', '变压器')
#             if type(row['设备类型']) != float:
#                 string = str(row['设备类型']).replace('主变', '变压器')
#                 if value in string or value in row['报告名称']:
#                     score = score + 20
#                 elif value in str(row['状态量']):
#                     score = score + 10
#                 else:
#                     score = score + 0.1 * fuzz.ratio(value, string)
#         if word_dict['故障'] != 'null':
#             score = score - 20
#             value = word_dict['故障'].replace('主变', '变压器')
#             if type(row['故障名称']) != float:
#                 string = str(row['故障名称']).replace('主变', '变压器')
#                 if value in string or value in row['报告名称']:
#                     score = score + 20
#
#         if word_dict['故障原因'] != 'null':
#             score = score - 10
#             value = word_dict['故障原因'].replace('主变', '变压器')
#             if type(row['故障名称']) != float:
#                 string = str(row['故障名称']).replace('主变', '变压器')
#                 if value in string or value in row['报告名称']:
#                     score = score + 10
#         if word_dict['电压等级'] != 'null':
#             nums2 = re.findall(r"\d+\.?\d*", str(row['电压等级']))
#             if nums2:
#                 num2 = nums2[0]
#                 num = str(word_dict['电压等级'])
#                 if num != num2:
#                     score = score - 10
#             else:
#                 score = score - 10
#         if word_dict['温度'] != 'null':
#             contain1 = '-' in str(row['测试环境温度'])
#             contain2 = '-' in str(word_dict['温度'])
#             if contain1 == contain2:
#                 nums2 = re.findall(r"\d+\.?\d*", str(row['测试环境温度']))
#                 if nums2:
#                     nums = re.findall(r"\d+\.?\d*", str(word_dict['温度']))
#                     num2 = int(nums2[0])
#                     num = int(nums[0])
#                     if num != num2:
#                         score = score - min(abs(num - num2), 10)
#                 else:
#                     score = score - 10
#             else:
#                 score = score - 10
#         if word_dict['湿度'] != 'null':
#             nums2 = re.findall(r"\d+\.?\d*", str(row['测试环境湿度']))
#             if nums2:
#                 num2 = int(nums2[0])
#                 num = int(word_dict['湿度'])
#                 if num != num2:
#                     score = score - min(abs(num - num2), 10)
#             else:
#                 score = score - 10
#
#         advice = '[' + row['检修决策'][1:-1].replace('"', "\\\"").replace('\'', '"') + ']'
#         analyse = row['诊断分析'].replace('\'', '"')
#         if is_json(advice) and row['诊断分析'] != '[]' and is_json(analyse) :
#
#             data_list.append({'设备类型':row['故障设备类型'],
#                                 '部件':row['状态量'],
#                                 '状态描述':row['状态描述'],
#                                 '诊断分析':analyse,
#                                 '检修决策':advice,
#                                 '匹配度':score})
#         # else:
#         #     print(advice)
#         #     print(row)
#
#
#     data_list.sort(key=lambda x:x['匹配度'], reverse=True)
#     json_result = json.dumps(data_list)
#     return json_result
#
#
# def sort_reports2(inputpath, json_str): #返回list
#     word_dict = json.loads(json_str)
#     data = pd.read_csv(inputpath, encoding='utf-8')
#     data_dict = {}
#     for index, row in data.iterrows():
#         score = 100
#         # for key, value in word_list.items():
#         if word_dict['设备类型'] != 'null':
#             value = word_dict['设备类型'].replace('主变', '变压器')
#             if type(row['故障设备类型']) != float:
#                 string = str(row['故障设备类型']).replace('主变', '变压器')
#                 if value not in string:
#                     score = score - 20
#             else:
#                 score = score - 20
#         if word_dict['部件'] != 'null':
#             score = score - 20
#             value = word_dict['部件'].replace('主变', '变压器')
#             if type(row['设备类型']) != float:
#                 string = str(row['设备类型']).replace('主变', '变压器')
#                 if value in string or value in row['报告名称']:
#                     score = score + 20
#                 elif value in str(row['状态量']):
#                     score = score + 10
#                 else:
#                     score = score + 0.1 * fuzz.ratio(value, string)
#         if word_dict['故障'] != 'null':
#             score = score - 20
#             value = word_dict['故障'].replace('主变', '变压器')
#             if type(row['故障名称']) != float:
#                 string = str(row['故障名称']).replace('主变', '变压器')
#                 if value in string or value in row['报告名称']:
#                     score = score + 20
#
#         if word_dict['故障原因'] != 'null':
#             score = score - 10
#             value = word_dict['故障原因'].replace('主变', '变压器')
#             if type(row['故障名称']) != float:
#                 string = str(row['故障名称']).replace('主变', '变压器')
#                 if value in string or value in row['报告名称']:
#                     score = score + 10
#         if word_dict['电压等级'] != 'null':
#             nums2 = re.findall(r"\d+\.?\d*", str(row['电压等级']))
#             if nums2:
#                 num2 = nums2[0]
#                 num = str(word_dict['电压等级'])
#                 if num != num2:
#                     score = score - 10
#             else:
#                 score = score - 10
#         if word_dict['温度'] != 'null':
#             contain1 = '-' in str(row['测试环境温度'])
#             contain2 = '-' in str(word_dict['温度'])
#             if contain1 == contain2:
#                 nums2 = re.findall(r"\d+\.?\d*", str(row['测试环境温度']))
#                 if nums2:
#                     nums = re.findall(r"\d+\.?\d*", str(word_dict['温度']))
#                     num2 = int(nums2[0])
#                     num = int(nums[0])
#                     if num != num2:
#                         score = score - min(abs(num - num2), 10)
#                 else:
#                     score = score - 10
#             else:
#                 score = score - 10
#         if word_dict['湿度'] != 'null':
#             nums2 = re.findall(r"\d+\.?\d*", str(row['测试环境湿度']))
#             if nums2:
#                 num2 = int(nums2[0])
#                 num = int(word_dict['湿度'])
#                 if num != num2:
#                     score = score - min(abs(num - num2), 10)
#             else:
#                 score = score - 10
#
#         key = row['报告名称']
#         data_dict[key] = score
#
#     sorted_list = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
#
#     newdata = []
#     heads = data.columns.tolist()
#     heads.append('匹配度')
#     newdata.append(heads)
#     for key, value in sorted_list[:20]:
#         for index2, row2 in data.iterrows():
#             if row2['报告名称'] == key:
#                 datalist = row2.tolist()
#                 datalist.append(int(value))
#                 for i in range(len(datalist)):
#                     if str(datalist[i]) == 'nan':
#                         datalist[i] = ''
#                 print(datalist)
#                 newdata.append(datalist)
#     return newdata
#
# def write_csv(path, data):
#     with open(path, 'w', encoding='utf-8-sig', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         for d in data:
#             csv_writer.writerow(d)
#
# if __name__ == '__main__':
#     # path = sys.argv[1]
#     # path1 = sys.argv[2]
#     # path = './data/result.csv'  # 再处理的输入路径
#     # path1 = './data/newresult1217v1.csv'  # 输出路径
#     path = r'./data/unsorted.csv'
#     path2 = r'./data/sorted.csv'
#     word_list = {'设备类型':'输电线路', '部件':'线夹' ,'故障':'跳闸','故障原因':'鸟害','电压等级':'220',  '温度':'null', '湿度':'35'}
#     json_str = json.dumps(word_list)
#     data = sort_reports(path, json_str)
#     print(data)
#     # write_csv(path2, data)
#
#     # data = pd.read_csv(path, encoding='utf-8')
#     # data2 = pd.read_csv(path1, encoding='utf-8')
#     # newdatalist = []
#     # with open(path2, 'w', encoding='utf-8-sig', newline='') as csv_file:
#     #     csv_writer = csv.writer(csv_file)
#     #     heads = data.columns.tolist()
#     #     heads.append('状态量')
#     #     heads.append('状态描述')
#     #     heads.append('诊断分析')
#     #     heads.append('检修决策')
#     #     print(heads)
#     #     csv_writer.writerow(heads)
#     #     for index, row in data.iterrows():
#     #         datalist = row.tolist()
#     #         for index2, row2 in data2.iterrows():
#     #             if row2['报告名称'] == row['报告名称']:
#     #                 datalist.append(row2['状态量'])
#     #                 datalist.append(row2['状态描述'])
#     #                 datalist.append(row2['诊断分析'])
#     #                 datalist.append(row2['检修决策'])
#     #         for i in range(len(datalist)):
#     #             if str(datalist[i]) == 'nan':
#     #                 datalist[i] = ''
#     #         print(datalist)
#     #         csv_writer.writerow(datalist)
#
