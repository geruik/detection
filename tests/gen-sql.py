"""
动态生成sql语句
"""
import random

PRODUCTS = [
    {"product_key": "zhinengchazuo", "device_type": "CHAZUO"},
    {"product_key": "huanjingjiankong", "device_type": "HUANJINGJIANKONG"},
    {"product_key": "wifisuo", "device_type": "WIFISUO"},
    {"product_key": "rwcard", "device_type": "DUKAQI"},
    {"product_key": "airswitch", "device_type": "KONGKAI"},
    {"product_key": "kehuduan", "device_type": "KEHUDUAN"},
    {"product_key": "lanaochazuo", "device_type": "CHAZUO"},
    {"product_key": "lorachazuo", "device_type": "CHAZUO"},
    {"product_key": "bluetoothchazuo", "device_type": "CHAZUO"},
    {"product_key": "loraenvironment", "device_type": "HUANJINGJIANKONG"},
    {"product_key": "lorarwcard", "device_type": "DUKAQI"},
    {"product_key": "lorasmoke", "device_type": "HUANJINGJIANKONG"},
    {"product_key": "haikangmenjin", "device_type": "MENJIN"},
    {"product_key": "zhinenggui", "device_type": "SHIJIGUI"},
    {"product_key": "zhinengtianping", "device_type": "TIANPING"},
    {"product_key": "zhinengshexiangtou", "device_type": "SHEXIANGTOU"},
    {"product_key": "zhinengdingwei", "device_type": "ZHINENGDINGWEI"},
    {"product_key": "lanyakongzhiqi", "device_type": "LANYA"},
    {"product_key": "peoplecounterwifi", "device_type": "PEOPLECOUNT"},
    {"product_key": "peoplecounterlora", "device_type": "PEOPLECOUNT"},
    {"product_key": "hikcamera", "device_type": "SHEXIANGTOU"},
    {"product_key": "dicard", "device_type": "DUKAQI"},
    {"product_key": "nanyibanpai", "device_type": "DIANZIBANPAI"},
    {"product_key": "yiqiclient", "device_type": "KEHUDUAN"},
    {"product_key": "hikthermal", "device_type": "SHEXIANGTOU"},
    {"product_key": "loraws502", "device_type": "CHAZUO"},
    {"product_key": "lorawt302", "device_type": "CHAZUO"},
    {"product_key": "empaergas", "device_type": "HUANJINGJIANKONG"},
    {"product_key": "empaerlel", "device_type": "HUANJINGJIANKONG"},
    {"product_key": "hikviolation", "device_type": "SHEXIANGTOU"},
]
"""设备产品类型类别"""


bianhao_idx = 556789

def pick_random_product():
    """随机选择一个产品"""
    return random.choice(PRODUCTS)


def gen_sql():
    """生成一条数据的插入SQL语句"""
    global bianhao_idx
    bianhao_idx+=1
    product = pick_random_product()
    leixing = product['device_type']
    product_key = product['product_key']
    sql = f'''
            INSERT INTO znyj_yingjian
        (
        bianhao,
        mingcheng,
        leixing,
        pinpai,
        guige,
        lianjie,
        qiyong,
        tupian,
        user_id,
        room_id,
        weizhi_leixing,
        weizhi_biaoshi,
        bind_id,
        beizhu,
        mac,
        created_time,
        updated_time,
        version,
        config,
        deleted,
        organize_id,
        bangding_leixing,
        bangdingrongqi,
        yeweishangxian,
        yeweixiaxian,
        wendushangxian,
        wenduxiaxian,
        zhidongjilushijian,
        yewei_yichang_tixing,
        wendu_yichang_tixing,
        menxinxi,
        account,
        password,
        yiqi_id,
        fuzeren_id,
        feifashiyongbaojing,
        gongzuodianliuyuzhi,
        is_warning,
        dianyashangxian,
        dianliushangxian,
        product_key,
        is_platform_create,
        shipinjietu,
        is_fire_alarm,
        business_ip,
        business_port,
        box_id
        )
        VALUES
        (
        '{bianhao_idx}',
        '{leixing}-{bianhao_idx}',
        '{leixing}',
        NULL,
        NULL,
        'YILIANJIE',
        'QIYONG',
        NULL,
        1,
        NULL,
        'FANGJIAN',
        NULL,
        NULL,
        NULL,
        NULL,
        '2024-06-14 15:45:02',
        '2024-07-22 17:15:01',
        0,
        NULL,
        0,
        1,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        0,
        0,
        'DANMEN',
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        0,
        NULL,
        NULL,
        '{product_key}',
        0,
        NULL,
        0,
        NULL,
        NULL,
        NULL
        );
    
    '''
    return sql

file= open('insert.sql', 'a') 
for i in range(138):
    file.write(gen_sql())