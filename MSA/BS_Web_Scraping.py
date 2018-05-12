# -*- coding: utf-8 -*-
#libraries
import bs4
from bs4 import BeautifulSoup
#import urllib2
from urllib.request import urlopen
import csv
import numpy 

out = open( 'C:/Users/William/Desktop/NCSU MSA/Fall 2015/Python/Web Scraping/temp.csv', 'wb' )
writer = csv.writer( out )

cities=[ "http://www.accuweather.com/en/bg/sofia/51097/july-weather/51097",
"http://www.accuweather.com/en/br/sao-paulo/45881/july-weather/45881",
"http://www.accuweather.com/en/ca/vancouver/v5y/july-weather/53286",
"http://www.accuweather.com/en/cn/chongqing/102144/july-weather/102144",
"http://www.accuweather.com/en/hr/zagreb/117910/july-weather/117910",
"http://www.accuweather.com/en/et/addis-ababa/126831/july-weather/126831",
"http://www.accuweather.com/en/eg/cairo/127164/july-weather/127164",
"http://www.accuweather.com/en/gh/accra/178551/july-weather/178551",
"http://www.accuweather.com/en/in/chennai/206671/july-weather/206671",
"http://www.accuweather.com/en/np/kathmandu/241809/july-weather/241809",
"http://www.accuweather.com/en/pk/karachi/261158/july-weather/261158",
"http://www.accuweather.com/en/ru/moscow/294021/july-weather/294021",
"http://www.accuweather.com/en/tw/taipei-city/315078/july-weather/315078",
"http://www.accuweather.com/en/ua/odessa/325343/july-weather/325343",
"http://www.accuweather.com/en/us/raleigh-nc/27601/july-weather/329823"]

# EXTRACT DATE AND WRITE HEADER ROW OF CSV FILE
url=cities[0]
page=urlopen(url)
soup = BeautifulSoup(page.read())
page.close()
#DATE
current_month=soup.find_all("td",{"class":"nf"})
cur_date=[]
for i in range(0,len(current_month)):
	cur_date.append(current_month[i].find("h3",{"class":"date"}))
date_list=[]
for i in range(0,len(cur_date)):
	date_list.append(cur_date[i].getText().encode("unicode_escape"))
#WRITING HEADER ROW OF CSV FILE
header=['city']
for i in range(0,len(date_list)):
	header.append(date_list[i])
header.append('avg')
writer.writerow(header)

#TEMPERATURE
for i in range(len(cities)):
	url=cities[i]
	page2=urlopen(url)
	tempsoup = BeautifulSoup(page2.read())
	page2.close()
	#CITY
	cityext = tempsoup.find( 'div', { 'id': 'country-settings' } )
	cityext1 = cityext.find( 'li', { 'class': 'last' } )
	city=cityext1.getText().encode("latin-1")
	#CURRENT TEMPERATURE
	current_month=[]
	current_month=tempsoup.find_all("td",{"class":"nf"})
	temp_month=[]
	for i in range(0,len(current_month)):
		temp_month.append(current_month[i].find("span",{"class":"temp"}))
	temp_list=[]
	for i in range(len(temp_month)):
		temp_list.append(temp_month[i].getText()[:-1].encode("unicode_escape"))
	#HISTORICAL TEMPERATURE	
	current_month2=tempsoup.find_all("td", {"class":"nf"})
	avg_month=[]
	for i in range(0,len(current_month2)):
		avg_month.append(current_month2[i].find("div",{"class":"avg"}))
	avg_month2=[]
	for i in range(0,len(avg_month)):
		avg_month2.append(avg_month[i].find("span",{"class":"temp"}))
	avg_list=[]
	for i in range(0,len(avg_month2)):
		if avg_month2[i].getText()[0].encode("unicode_escape")=='N':
			avg_list.append('17')
		else:
			avg_list.append(avg_month2[i].getText()[:-1].encode("unicode_escape"))
		
	diff=[]
	for i in range(0,len(temp_list)):
		diff.append(int(temp_list[i]) - int(avg_list[i]))
		
	avg_diff=format(numpy.mean(diff),'.2f')

	nextrow=[]
	nextrow=[city]
	for i in range(0,len(diff)):
		nextrow.append(diff[i])
	nextrow.append(avg_diff)
	writer.writerow(nextrow)

out.close()
