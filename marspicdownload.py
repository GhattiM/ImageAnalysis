# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:14:02 2016

@author: rohit
"""
import urllib
import urllib2
import json
with open ("E:\\bigdata\\project\\marspics.json") as mpics:
    d = json.load(mpics)
    print(d)
#result = json.load(url)    
url1 = 'https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?sol=1000&api_key=ZUBprhtzffMvYVHgP1egFHFaNI6jq88bDEgNbvzm'
url='https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?sol=1000&page=5&api_key=ZUBprhtzffMvYVHgP1egFHFaNI6jq88bDEgNbvzm'
req = urllib2.Request(url)
resp = urllib2.urlopen(req)
json_str = json.load(resp)
i=len(json_str['photos'])
        
j=0    
while j<i:
    lst = str(json_str['photos'][j]['img_src']).split('/',len(str(json_str['photos'][j]['img_src'])))
    urllib.urlretrieve(json_str['photos'][j]['img_src'],"E:\\bigdata\\project\\marspic\\" + lst[len(lst)-1])
    j+=1
    
x=900
while x < 1000:
    url='https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?sol='+str(x)+'&api_key=ZUBprhtzffMvYVHgP1egFHFaNI6jq88bDEgNbvzm'
    req = urllib2.Request(url)
    resp = urllib2.urlopen(req)
    json_str = json.load(resp)
    i=len(json_str['photos'])
    print i
    j=0
    while j<i:
        lst = str(json_str['photos'][j]['img_src']).split('/',len(str(json_str['photos'][j]['img_src'])))
        urllib.urlretrieve(json_str['photos'][j]['img_src'],"E:\\bigdata\\project\\marspic2\\" + lst[len(lst)-1])
        j+=1
    x+=1


 
   #urllib.urlretrieve(json_str['photos'][0]['img_src'],"E:\\bigdata\\project\\marspic\\" + str(json_str['photos'][0]['img_src']).split('/',len(str(json_str['photos'][0]['img_src'])))[len(json_str['photos'][0]['img_src'])-1])