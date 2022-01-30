# Conditions-for-biodiversity

## [Link to the report](./Lullington/report/Lullington_Heath_report2.pdf)

### How I started this project

I have loved the natural world since learning about it in my childhood from my mother. It is an almost infintely complicated system, built over millions of years to a deep and stable balance. This balance is built on diversity and is crucial to how the world around us functions. We need insects to pollinate our crops, and for insects to be diverse and abundant, we need wild ecosystems with a variety of flowering plants. We need the ocean ecosystems to ensure large populations of plankton which sequester carbon and help to prevent climate change. Naturally I have a passion for ecology and conservation but what I wanted was a way to make tangible contributions to maintaining that balance and protecting the natural world.

[Natural England](https://www.gov.uk/government/organisations/natural-england) is the UK Government's official environmental advisory body, responsible for the designation and protection of the UK's wildlife. Conservation requires an enormous amount of data collection in order to monitor long-term changes to habitats and so I knew that there would be extensive survey data being collected by Natural England. I was curious what this data looked like, what was being done with it and was eager to see if I could come up with anything useful. I inquired and found they have a project called [Long-term Monitoring Network](http://publications.naturalengland.org.uk/publication/4654364897050624) (LTMN), under which they have been collecting data on sites all over the south-east of England for over ten years but as of yet, very little has been done with this data. I contacted the LTMN team and registered as an official volunteer, ready to see if I could find a way to make a contribution to the fight for the environment.

### Background information

Across the UK, there are a large number of Sites of Special Scientific Interest (SSSI) which have been designated as important areas of nature for either geological or biological reasons. It is Natural England's role to manage these sites to maintain the reason they were assigned as SSSI. This project will deal only with the biologically important SSSIs and as such we will only be referring to these sites in future. The reason for the designation could be the presence of a specific rare butterfly or lizard; it could be that the habitat is rare or declining; it could be a particularly high plant biodiversity (number of species per area). The sites are managed in various ways such as influencing nearby farming practices, grazing with livestock, volunteer manual labor etc but it is essential that a range of indicators are recorded to determine if the management practices are working. To this end Natural England survey these sites every ~4 years, taking ~50 plots in each and recording physical characteristics and species present in a 2m x 2m square.

### Aims for this project

##### [Code to clean a variety of surveys and formulate them into a consistent form](./clean.py)

The data collected is from a huge number of workers across a large timespan and as such there are many discrepancies across each plot, survey, site. In addition to this there are many typos, erroneously input data and spelling mistakes. In addition it was collected by people who were not experienced with handling data and so is not in a form ready for analysis. Most importantly though, no-one on the Natural England LTMN team knew how to code, so this would be an incredibly laborious process in excel for them. I realised that probably the single most useful thing I could do would be to write a suite of programs that could take in any survey data, clean it without specific modifications for each survey and present it in a consistent form.

##### Code to automatically generate figures for each survey and site

The team were previously using excel to write reports on individual surveys, with over 80 sites and many surveys for each this was slow progress and the vast majority of sites had no data analysis done. I wrote code as general as possible to generate tables of pertinent data and figures which could be used to fast-track report writing for the team.

##### [Write a report on the changing ecosystem of a site using the LTMN data](./Lullington/report/Lullington_Heath_report2.pdf)

I proposed writing a report on a local site because I wanted to finally turn this wealth of data into something useful for the organisation. I contacted the site manager at Lullington Heath and told them about my work with LTMN. They were very interested and we held a meeting to discuss would would be most useful. The resulting report was given a week later and has since been used to make changes to the way Lullington Heath is managed.

I was able to find key indications of a decline in habitat health where they had not yet been identified. This enabled Natural England to make swift changes to habitat management policy with the aim to reverse these changes.

##### Further work

As the data is now, it is very awkward to compare different sites to each other. While they are all different, many contain the same types of habitats and I have proposed that a comparison would be informative. To this end I would like to create a relational database with all current data and create code for the team to automatically add new survey data to the database

I also have some ideas for some machine learning mini-projects. I want to use the available data to categorise the habitats, firstly in a supervised manor according to the habitats recognised by Natural England. In this way I can see if certain habitats are declining over time. I will also see if certain characteristics of a habitat are in decline. 

Next in an unsupervised manor as the pre-designated habitats are old and are determined by biological indicators without the aid of machine learning. I want to see how well unsupervised methods match the biological expectations and potentially look for any new emerging habitats. 

I also want to determine which parts of the recorded data are most important in determining the quality of the SSSI. The recording of this data takes a huge amount of time from skilled workers. If I can point towards the key elements of the recorded data, I can improve efficiency and save Natural England a lot of money in the form of skilled labor.
