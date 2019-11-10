$(document).ready(function () {
//    var baseurl = "http://localhost:8085";
//    var report_id = window.location.pathname.match(/[0-9]+/)[0];
//    console.log(baseurl);
//    console.log(report_id);

    $("#btn").click(function () {
        $("#tableSpace").html('');
    });
    $('#content').css('min-height', window.innerHeight);
    $('iframe').css('height', window.innerHeight - 112);
    $('#export_content').css('height', window.innerHeight - 155);
    $(window).resize(function () {
        $('#content').css('min-height', window.innerHeight);
        $('iframe').css('height', window.innerHeight - 112);
        $('#export_content').css('height', window.innerHeight - 155);
    });
    $("#load").click(function () {
        $.ajax({
            type: "GET",
            url: baseurl + "/report/getPics",
            contentType: "application/json; charset=utf-8",
            data: {
                id: report.report_id
            },
            dataType: "json",
            success: function (response) {
                console.log(response);
                var shtml = "";
                for (var i = 0; i < response[0].pics.length; i++) {
                    shtml += "<img src='" + baseurl + "/pics/" + response[0].pics[i] + "' width='90%'/>";
                    $('#imgSpace').html("");
                    $('#imgSpace').html(shtml);
                }
            },
            error: function (message) {
                alert("提交数据失败！");
            }
        });
    });

    $("#save").click(function () {
        var reportNew = {};
        multi = ['#type_code', '#tower_code', '#extra_time', '#err_how'];
        id = "";
        inner = "";
        $("tr").each(function () {
            tds = $(this).children('td');
            if (tds.length < 2 && tds.attr('id') != 'imgSpace') {
                if (tds.text() != "")
                    reportNew[tds.attr("id")] = tds.text();
                else reportNew[tds.attr("id")] = null;
            } else if (tds.length == 2) {
                key = $(this).children('td:first');
                value = $(this).children('td:eq(-1)');
                if (key.is('[id]')) {
                    id = key.attr("id");
                    inner = "";
                }
                console.log(key.text()!="请在此添加");
                if(key.text()!=""&&key.text()!="请在此添加"){
                    inner += "(" + key.text() + "," + value.text() + ")、";
                }
                if(inner!=""){
                    reportNew[id] = inner;
                }
                else{
                    reportNew[id] = null;
                }
                
            }
        })
        reportNew["report_id"] = report.report_id;
        console.log(reportNew);
        console.log(JSON.stringify(reportNew));
        $.ajax({
            type: "Post",
            url: baseurl + "/report/updateReport",
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify(reportNew),
            dataType: "text",
            success: function (response) {
                console.log(response);
                var shtml = "";
            },
            error: function (message) {
                alert("提交数据失败！");
            }
        });
    });
    $("#export").click(function () {
        alert("hello");
        $.ajax({
            type: "get",
            url: baseurl + "/report/getPDF",
            contentType: "application/json; charset=utf-8",
            data: {
                id: report.report_id
            },
            dataType: "json",
            success: function (response) {
                console.log(response);
                var $aa = $('<a>');
                $aa.attr('href', baseurl + response.url);
                if ($.browser.msie || $.browser.mozilla) {
                    $aa.get(0).click();
                } else {
                    window.open($aa.attr('href'), $aa.attr('target'));
                }
            },
            error: function (message) {
                alert("提交数据失败！");
            }
        });
    });

//    
//    var generateData = function (amount) {
//        var result = [];
//        var data =
//        {
//            coin: "100",
//            game_group: "GameGroup",
//            game_name: "XPTO2",
//            game_version: "25",
//            machine: "20485861",
//            vlt: "0"
//        };
//        for (var i = 0; i < amount; i += 1) {
//            data.id = (i + 1).toString();
//            result.push(Object.assign({}, data));
//        }
//        return result;
//    };
//
//    function createHeaders(keys) {
//        var result = [];
//        for (var i = 0; i < keys.length; i += 1) {
//            result.push({
//            'id' : keys[i],
//                'name': keys[i],
//                'prompt': keys[i],
//                'width': 65,
//                'align': 'center',
//                'padding': 0
//            });
//        }
//        return result;
//    }
//
//
//
//    var headers = createHeaders(["id", "coin", "game_group", "game_name", "game_version", "machine", "vlt"]);
//
//    var doc = new jsPDF({ putOnlyUsedFonts: true, orientation: 'landscape' });
//    doc.table(1, 1, generateData(100), headers, { autoSize: true });

    function parseMulti(id) {
        parsed = "";
        reg = /\([^,]+,[^,]+\)/g
        ll = $(id).text().match(reg);
        if ($(id).text()=="") {
            $(id).siblings('th').attr('rowspan', 1);
            $(id).text('请在此添加');
            $(id).after('<td class="short-text">' + '请在此添加' + '</td>');
        } else {
            for (var i in ll) {
                text = ll[i].substr(1, ll[i].length - 2).split(',');
                if (i == 0) {
                    $(id).text(text[0]);
                    $(id).after('<td class="short-text">' + text[1] + '</td>');
                } else {
                    parsed += '<tr><td class="short-text">' + text[0] + '</td>:<td class="short-text">' + text[1] + '</td></tr>';
                }
            }
            $(id).siblings('th').attr('rowspan', ll.length);
            $(id).parents('tr').after(parsed);
        }
    };

    function arrangeTable() {
        multi = ['#type_code', '#tower_code', '#extra_time', '#err_how'];
        for (i in multi) {
            parseMulti(multi[i]);
        }
        $("tr").each(function () {
            if ($(this).children('td').length < 2) {
                $(this).children('td').attr('colSpan', 2);
            }
        })
    }
    arrangeTable();
    $('.btn-add').click(function () {
        father = $(this).parents("th");
        x = parseInt(father.attr('rowspan')) + 1;
        console.log(parseInt(father.attr('rowspan')) + 1);
        text = [];
        $(this).parents('th').siblings('td').each(function () {
            text.push($(this).text());
        });
        $(this).parents('th').siblings('td').text('请在此添加');
        $(this).parents('tr').after('<tr><td class="short-text">' + text[0] + '</td><td class="short-text">' + text[1] + '</td></tr>');
        $(this).parents('th').attr('rowSpan', x);

    });

    $('body').on('click', ".short-text", function (event) {
        //td中已经有了input,则不需要响应点击事件
        if ($(this).children("input").length > 0)
            return false;
        if ($('#tableSpace input').length > 0) {
            var preInput = $('input');
            preInput.parent('td').html(preInput.val());
        }
        if ($('#tableSpace textarea').length > 0) {
            var preInput = $('textarea');
            preInput.parent('td').html(preInput.val());
        }
        var tdObj = $(this);
        var preText = tdObj.html();
        var preHeight = tdObj.height();
        var preWidth = tdObj.width();
        //得到当前文本内容
        var inputObj = $("<input type='text' />");
        //创建一个文本框元素
        tdObj.html(""); //清空td中的所有元素
        inputObj.width(preWidth)
            //设置文本框宽度与td相同
            .height(preHeight)
            .css({
                border: "0px",
                fontSize: "17px",
                font: "宋体"
            })
            .val(preText)
            .appendTo(tdObj)
            //把创建的文本框插入到tdObj子节点的最后
            .trigger("focus")
            //用trigger方法触发事件
            .trigger("select");
        inputObj.keyup(function (event) {
            if (13 == event.which)
            //用户按下回车
            {
                var text = $(this).val();
                tdObj.html(text);
            } else if (27 == event.which)
            //ESC键
            {
                tdObj.html(preText);
            }
        });
        //已进入编辑状态后，不再处理click事件
        inputObj.click(function () {
            return false;
        });
    });
    $(".long-text").click(function (event) {
        //td中已经有了input,则不需要响应点击事件

        if ($(this).children("textarea").length > 0)
            return false;
        if ($('#tableSpace input').length > 0) {
            var preInput = $('input');
            preInput.parent('td').html(preInput.val());
        }
        if ($('#tableSpace textarea').length > 0) {
            var preInput = $('textarea');
            preInput.parent('td').html(preInput.val());
        }
        var tdObj = $(this);
        var preText = tdObj.html();
        var preHeight = tdObj.height();
        var preWidth = tdObj.width();
        if (preHeight < 100)
            preHeight = 100;
        //得到当前文本内容
        var inputObj = $("<textarea/>")

        //创建一个文本框元素
        tdObj.html(""); //清空td中的所有元素
        inputObj.width(preWidth)
            //设置文本框宽度与td相同
            .height(preHeight)
            .css({
                border: "0px",
                fontSize: "17px",
                font: "宋体"
            })
            .val(preText)
            .appendTo(tdObj)
            //把创建的文本框插入到tdObj子节点的最后
            .trigger("focus")
            //用trigger方法触发事件
            .trigger("select");
        inputObj.keyup(function (event) {
            if (13 == event.which)
            //用户按下回车
            {
                var text = $(this).val();
                tdObj.html(text);
            } else if (27 == event.which)
            //ESC键
            {
                tdObj.html(preText);
            }
        });
        //已进入编辑状态后，不再处理click事件
        inputObj.click(function () {
            return false;
        });
    });
});
