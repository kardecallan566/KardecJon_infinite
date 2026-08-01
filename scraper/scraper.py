import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import random

class Scraper:
    def __init__(self, url: str):
        self.url = url
        self.logger = logging.getLogger(__name__)

    def fetch_latest_result(self, mock: bool = False):
        """
        Fetches the latest result from the website.
        If mock=True, returns a random result for testing.
        """
        if mock:
            return self._get_mock_result()
        
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Based on the user's provided HTML structure:
            # <div class="history__double__container">
            #   <div class="history__double__item history__double__item--red">
            #     <div class="history__double__center">2</div>
            #   </div>
            #   <div class="history__double__date"><p>01/08/2026</p><p>20:28:11</p></div>
            # </div>
            
            container = soup.find('div', class_='history__double__container')
            if not container:
                self.logger.warning("Container not found in HTML")
                return None
            
            item = container.find('div', class_='history__double__item')
            color_class = item.get('class', [])
            
            # Map classes to colors
            color = "BRANCO"
            if "history__double__item--red" in color_class:
                color = "VERMELHO" # The user said "Preto e Verde", but standard is Red/Black/White.
                # Adjusting to user's request: Preto (47%), Verde (47%), Branco (6%)
                # I'll map Red -> Verde and Black -> Preto based on common patterns or just use user's names.
                color = "VERDE"
            elif "history__double__item--black" in color_class:
                color = "PRETO"
            
            number = int(item.find('div', class_='history__double__center').text.strip())
            
            date_div = container.find('div', class_='history__double__date')
            ps = date_div.find_all('p')
            date_str = ps[0].text.strip()
            time_str = ps[1].text.strip()
            
            return {
                "color": color,
                "number": number,
                "date": date_str,
                "time": time_str
            }
            
        except Exception as e:
            self.logger.error(f"Error scraping data: {e}")
            return None

    def _get_mock_result(self):
        rand = random.random()
        if rand < 0.06:
            color = "BRANCO"
        elif rand < 0.53:
            color = "PRETO"
        else:
            color = "VERDE"
            
        return {
            "color": color,
            "number": random.randint(0, 14),
            "date": datetime.now().strftime("%d/%m/%Y"),
            "time": datetime.now().strftime("%H:%M:%S")
        }
