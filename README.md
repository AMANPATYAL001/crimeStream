# Manthan 2021: Team SecureTech WebApp (Version 1)

### This a part of **[Manthan 2021](https://manthan.mic.gov.in/about-intellithon.php)** (Hackathon for National Security).
<br>

**[Link for WebApp](http://securetech.herokuapp.com/)**
<br>

**[Full project Description Link](https://github.com/Manukumar9319/MANTHAN_2021_SecureTech)**
<br>

![](res/pic1.png)
Map of north-eastern Lucknow(provided by Manthan, with each point representing accidents or crimes (click on points for more info.).
<br><br>

![](res/setting.png)
### User options like:
- No. of tweets to examin
- location name
- lat, long values
- radius of the region 

<br><br>

![](res/pic2.png)

### List of user info with negative Tweets **(Sentiment model is not 100% accurate, so please consider tweets on your own risk)**, wordcloud and stacked bar graph with no. of positive, neutral and negative tweets. 
<br><br>

### The model we used here is of TextBlob. We also created a sentiment analysis model(with accuracy 79%, LSTM layer, epoch=12, and training dataset is from kaggle(1.6M records)), but due to its size (350MB), it's not possible to deploy it on heroku.