"""Tiny built-in dataset so the project runs out of the box.
It is a TOY set for demo only. For real results, put ISOT (Fake.csv + True.csv)
or a news.csv (columns: text,label) inside /data and re-run train.py."""
FAKE = [
 "SHOCKING: Scientists confirm drinking hot water cures all cancers, doctors hate this!",
 "BREAKING!!! Government secretly microchipping citizens through new vaccine, leaked documents reveal",
 "You won't believe what this celebrity did, the media is hiding the TRUTH from you",
 "Miracle pill melts belly fat overnight without diet or exercise, big pharma furious",
 "Aliens landed in capital city and officials are covering it up, anonymous insider claims",
 "Share before it gets deleted: banks will freeze all accounts tomorrow, everyone must withdraw cash now",
 "Eating lemon and garlic daily makes you immune to every virus, shocking study they banned",
 "Secret elite cabal controls the weather to destroy crops, whistleblower exposes shocking plot",
 "Free smartphones for everyone from government, click now to claim before midnight!!!",
 "Famous politician arrested overnight, mainstream media refuses to report this explosive story",
 "Moon landing footage proven fake by new viral video that experts are afraid to discuss",
 "Drinking bleach water in the morning detoxifies the body completely, natural cure exposed",
]
REAL = [
 "Central bank keeps interest rates unchanged, citing stable inflation and steady economic growth",
 "Parliament passes budget bill after two days of debate, opposition members stage walkout",
 "Researchers publish peer-reviewed study on coastal erosion in the journal Nature Climate Change",
 "State government announces new metro line extension, construction to begin next financial year",
 "Health ministry reports decline in seasonal flu cases as vaccination drive reaches rural districts",
 "Stock markets close higher on Tuesday led by gains in banking and information technology shares",
 "Heavy rainfall alert issued for coastal districts as meteorological department tracks low pressure system",
 "University announces revised examination schedule following approval from the academic council",
 "Court dismisses petition challenging the land acquisition order, says procedure was followed",
 "Indian space agency successfully launches communication satellite from Sriharikota spaceport",
 "Local officials inaugurate new primary health centre, say it will serve around forty villages",
 "Trade delegation signs agreement to boost exports of textiles and agricultural products next year",
]
SEED = [(t, 0) for t in FAKE] + [(t, 1) for t in REAL]
