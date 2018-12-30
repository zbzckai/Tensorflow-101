# -*- coding: utf-8 -*-
# @Time    : 2018\12\28 0028 17:11
# @Author  : 凯
# @File    : MP4.py
# -*- coding: utf-8 -*-
# @Time    : 2018\12\28 0028 15:22
# @Author  : 凯
# @File    : MP4.py
import requests
response = requests.get("https://video.pearvideo.com/mp4/adshort/20181228/cont-1498351-13416439_adpkg-ad_hd.mp4")
with open('你是谁.mp4',mode='wb') as f:
    f.write(response.content)
###获取视频的真实地址
import re
info_url = 'https://www.pearvideo.com/video_1498351'
response = requests.get(info_url)
response.content.decode()##方法一
response.text## 方法二
import re
##.匹配任意字符*匹配一个或多个字符 ？ 非贪婪
re.findall('srcUrl="(.*?)"',response.text)##括号相当于取模块不然就都取出来了
MP4_url = re.search('srcUrl="(.*?)"',response.text)##查到第一个就停止了
print(MP4_url.group(1))
##查找名字
MP4_NAME = re.search('data-title="(.*?)"',response.text)
MP4_NAME.group(1)
def aa():
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaa')
if __name__ == 'main':
    print("shhhhh")