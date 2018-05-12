from bs4 import BeautifulSoup
import urllib2
import requests

url='http://www.accuweather.com/en/tw/taipei-city/315078/july-weather/315078'
r=requests.get(url)
data=r.text

soup=BeautifulSoup(data,"lxml")

#pos_day=soup.find("td")

for day in soup.find_all("td"):
    if soup.find("td",{"class":"nf"}):

        current_month=soup.find("td",{"class":"nf"})
#print current_month

        date=current_month.find("h3",{"class":"date"})
        print date.text

#for day in soup.find_all("div"):
 #   print(day.get("div"))
    
        actual=current_month.find("div",{"class":"actual"})
#print div
        a_temp=actual.find("span",{"class":"temp"})
        print a_temp.text[:-1]

        historic=current_month.find("div",{"class":"avg"})
#print historic
        h_temp=historic.find("span",{"class":"temp"})
        print h_temp.text[:-1]
        

#temp_pos=span.find(">")

#temp=span.find(">",2)
#print temp

#temp=span.gettext()
#print temp