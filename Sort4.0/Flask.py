from flask import Flask, request, jsonify
import json
import Sort
import re
from flask_cors import *  # 导入模块
from Algorithm.multi_label.advice import test_advice
from Algorithm.err_class.err import predict_err
from Algorithm.source_example.run import example_push
app = Flask(__name__)
app.debug = True


# @app.route('/', methods=['get'])
# def start():
#     # if not request.data:  # 检测是否有数据
#     #     return ('fail')
#     # json_str = request.args.get('callback') # 获取到POST过来的数据
#     # print(json_str)
#     params = {"设备类型":"null","部件":"null","故障":"null","故障原因":"null","湿度":"null","温度":"null","电压等级":"null"}
#     for k in params:
#         params[k] = request.args.get(k)
#     json_str = json.dumps(params)
#     json_data = Sort.sort_reports(inputpath='./data/unsorted.csv', json_str=json_str)
#     # resp = Response(json_data)
#     # resp.headers['Access-Control-Allow-Origin'] = '*'
#     return 'successCallback(%s)' % json_data # 返回JSON数据。
@app.route("/fault",methods = ['post'])
def fault_detect():
    data = request.get_data()

    json_data = json.loads(data.decode("utf-8"))
    test_content = json_data["desc"]
    ans = predict_err(test_content)
    # print(test_content)
    res = {"succeed":1,"message":"success","data":ans}
    return jsonify(res)


@app.route("/advice",methods = ['post'])
def advice_ai():
    data = request.get_data()

    json_data = json.loads(data.decode("utf-8"))
    test_content = json_data["desc"]
    st = "1.案例经过"
    ed = "4.检测相关信息"
    si = test_content.find(st)
    ei = test_content.find(ed)
    print("awsl",si,ei,test_content)
    if(si != -1 or ei != -1):
        test_content = test_content[si:ei]
    ans = test_advice(test_content)
    # print(test_content)
    res = {"succeed":1,"message":"success","data":ans}
    return jsonify(res)



@app.route("/source_example",methods = ['post'])
def source_example_ai():
    data = request.get_data()

    json_data = json.loads(data.decode("utf-8"))
    test_content = json_data["desc"]

    print("awsl",test_content)
    ans = example_push(test_content)
    print(ans)
    res = {"succeed":1,"message":"success","data":json.dumps(ans)}
    return jsonify(res)


if __name__ == '__main__':
    CORS(app, supports_credentials=True)  # 设置跨域
    app.run(host='127.0.0.1', port=8088) # 这里指定了地址和端口号。