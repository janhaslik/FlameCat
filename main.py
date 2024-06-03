from model import predict
from model import model
from pathlib import Path

model_file = Path('model/saved_model/flamecat-model.pt')

if model_file.is_file():
    res = predict.predict("""New Year's texting breaks record
 
 A mobile phone was as essential to the recent New Year's festivities as a party mood and Auld Lang Syne, if the number of text messages sent is anything to go by.
 
 Between midnight on 31 December and midnight on 1 January, 133m text messages were sent in the UK. It is the highest ever daily total recorded by the Mobile Data Association (MDA). It represents an increase of 20% on last year's figures.
 
 Wishing a Happy New Year to friends and family via text message has become a staple ingredient of the year's largest party. While texting has not quite overtaken the old-fashioned phone call, it is heading that way, said Mike Short, chairman of the MDA. ""In the case of a New Years Eve party, texting is useful if you are unable to speak or hear because of a noisy background,"" he said. There were also lots of messages sent internationally, where different time zones made traditional calls unfeasible, he said. The British love affair with texting shows no signs of abating and the annual total for 2004 is set to exceed 25bn, according to MDA. The MDA predicts that 2005 could see more than 30bn text messages sent in the UK. ""We thought texting might slow down as MMS took off but we have seen no sign of that,"" said Mr Short. More and more firms are seeing the value in mobile marketing. Restaurants are using text messages to tell customers about special offers and promotions.
 
 Anyone in need of a bit of January cheer now the party season is over, can use a service set up by Jongleurs comedy club, which will text them a joke a day. For those still wanting to drink and be merry as the long days of winter draw in, the Good Pub Guide offers a service giving the location and address of their nearest recommended pub. Users need to text the word GOODPUB to 85130. If they want to turn the evening into a pub crawl, they simply text the word NEXT. And for those still standing at the end of the night, a taxi service in London is available via text, which will locate the nearest available black cab.
""")
    print(res)
else:
    trainer = model.ModelTrainer()

