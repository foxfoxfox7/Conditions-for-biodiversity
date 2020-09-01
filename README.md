# Conditions-for-biodiversity

#### How I started this project

For this project I contacted a friend of mine who works for [Natural England](https://www.gov.uk/government/organisations/natural-england): the government's environmental agency. I knew that ecology and environmental protection requires an enormous amount of data collection and recording to give an indication of any changes in the habitats over time. I inquired and found they have a project called [Long-term Monitoring Network](http://publications.naturalengland.org.uk/publication/4654364897050624) (LTMN), under which they have been collecting data on sites all over the south-east of England for over ten years but as of yet, very little has been done with this data. I contacted the LTMN team, registered as an official volunteer and am currently undergoing an analysis of the data they have collected whilst liaising with the environment officers in charge of the sites that have been monitored.

#### Background information

Across the UK, there are a large number of Sites of Special Scientific Interest (SSSI) which have been designated as important areas of nature for either geological or biological reasons. It is Natural England's job to manage these sites to maintain the reason they were assigned as SSSI. This project will deal only with the biologically important SSSIs and as such we will only be referring to these sites in future. The reason could be the presence of a specific rare butterfly or lizard; it could be that the habitat is rare; it could be a particularly high plant biodiversity (number of species per area). They are managed in various ways such as influencing nearby farming practices, grazing with livestock, volunteer manual labor etc but it is essential that a range of indicators are recorded to determine if the management practices are working. To this end Natural England record these sites every ~4 years, taking ~50 points in each and recording physical characteristics and species present in a 2m x 2m square.

#### Aims for this project

The data collected is from a huge number of volunteers across a large timespan and as such there are many discrepancies across each plot, survey, site. In addition to this there are many typos, erroneously input data and spelling mistakes. My first aim is therefore to write code to clean this data and present it in a manageable format for data analysis. I was also asked to write it as general as possible so that it could be used for all future surveys as well.

- [x] Write code to clean all surveys that can be used by the organisation for all future surveys

I was tasked with an analysis of three sites in particular (out of 80). These are three completely different sites with different habitats but each is up for evaluation soon. My analysis will consist primarily of a set of figures showing how the conditions change over time. This includes biodiversity, pH, light, plant height amongst others. There are also a number of site specific requirements which I will include.

- [ ] A report on Dark Peak
- [ ] A report on Saltfleetby
- [ ] A report on Lullington

I want the code to be written in such a way as to make it very simple to write similar reports on all 80 sites with only small changes for the differences in specifications for each.

As the data is now, it is very awkward to compare different sites to each other. While they are all different, many contain the same types of habitats and I have proposed that a comparison would be informative. To this end I will create a relational database.

- [ ] Create a relational database from the collection of excel spreadsheets currently being used

I also have some ideas for some machine learning mini-projects: I want to use the available data to categorise the habitats. In this way I can see if certain habitats are declineing over time. I will also see if certain characteristics of a habitat are in decline. I also want to determine which parts of the recorded data are most important in determining the quality of the SSSI. The recording of this data takes a huge amount of time from skilled workers. If I can point towards a more streamlined way of recording data, I can save Natural England a lot of money.
