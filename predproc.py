import requests
from bs4 import BeautifulSoup
import pandas as pd



signs=['aries','taurus','taurus','cancer','leo','virgo','libra','scorpio','sagittarius','capricorn','aquarius','pisces']

for i in range(2004,2021):
    for j in range(1,13):
        for k in range(1,32):
            for sign in signs:
                try:
                    year=i
                    month=j
                    day=k
                    if month<10:
                        month='0'+str(month)
                    if day<10:
                        day='0'+str(day)
                    date='%s-%s-%s'%(year,month,day)
                    url = 'https://horoscopes.rambler.ru/%s/%s/?updated'% (sign,date)  # url для второй страницы
                    print(url)
                    r = requests.get(url)
                    response = r.text.encode('utf-8')

                    soup = BeautifulSoup(response, features="lxml")
                    text = soup.find('div', {'class': '_1dQ3'}).text
                    data = pd.read_csv('data.csv', names=['date', 'sign', 'text'])
                    data=data.append({'date': date, 'sign': sign, 'text': text}, ignore_index=True)
                    data.to_csv('data.csv', encoding = 'utf-8')
                except Exception:
                    continue