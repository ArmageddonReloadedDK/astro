import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import nest_asyncio
nest_asyncio.apply()


start=[]
end=[]
N=50

step=int(3000000/N)
print('step ',step)
for i in range(N):
    start.append(i*step)
    end.append((i+1)*step)


async def get_news(start, end):
    global news
    for i in range(start, end):
        url = 'https://www.1rnd.ru/news/{0}'.format(str(i))


        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:

                response = await response.text()
                soup = BeautifulSoup(response, 'html.parser')

                news_text = soup.findAll('p', {'style': "text-align:justify;"})
                date_text = soup.findAll('div', {'class': "article-info__time"})
                header_text = soup.findAll('div', {'class': 'title-container inner-title'})

                if news_text:
                    print('success')
                    print(url)
                    text_line = ''
                    for text in news_text:
                        text_line = text_line + text.get_text()

                    date = date_text[0].get_text()
                    header = header_text[0].get_text()

                    news = news.append({'header': header, 'date': date, 'news': text_line}, ignore_index=True)
                    news.to_csv('news_fast.csv', encoding='utf-8')

news = pd.DataFrame({'header': [0], 'date': [0], 'news': [0]})

futures = [get_news(start[i],end[i])for i in range(N)]

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(futures))
loop.close()