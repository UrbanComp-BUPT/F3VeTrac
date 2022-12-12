import math


def dis_cal(lon1, lat1, lon2, lat2):
    """
    计算A,B两点间的地球距离
    -------
    params:
    lon1: 点A的经度
    lat1: 点A的纬度
    lon2: 点B的经度
    lat2: 点B的纬度
    -------
    return:
    earth_dis: A,B两点间的地球距离
    """
    R = 6372800  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    
    # print(2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
    earth_dis = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_dis