import re
import urllib.request

url="http://image.baidu.com/search/index?ct=&z=&tn=baiduimage&word=%E8%BD%A6&pn=0&ie=utf-8&oe=utf-8&cl=&lm=-1&fr=&se=&sme=&width=1024&height=768"

request=urllib.request.Request(url)

response=urllib.request.urlopen(request)

data=response.read()

data=data.decode('UTF-8')


print(data)