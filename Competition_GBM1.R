NewsTrain = read.csv("NYTimesBlogTrain1.csv", stringsAsFactors=FALSE)

NewsTest = read.csv("NYTimesBlogTest1.csv", stringsAsFactors=FALSE)

str(NewsTrain)

# Now, let's load the "tm" package

library(stringr)

NewsTrain$combcol=str_c(NewsTrain$Headline,NewsTrain$Snippet,NewsTrain$Abstract)

NewsTest$combcol=str_c(NewsTest$Headline,NewsTest$Snippet,NewsTest$Abstract)

library(tm)

# Then create a corpus from the headline variable. You can use other variables in the dataset for text analytics, but we will just show you how to use this particular variable. 
# Note that we are creating a corpus out of the training and testing data.

CorpusHeadline = Corpus(VectorSource(c(NewsTrain$combcol, NewsTest$combcol)))

# You can go through all of the standard pre-processing steps like we did in Unit 5:

CorpusHeadline = tm_map(CorpusHeadline, tolower)

# Remember this extra line is needed after running the tolower step:

CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)

CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)

CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))

CorpusHeadline = tm_map(CorpusHeadline, stemDocument,language="english")

# Now we are ready to convert our corpus to a DocumentTermMatrix, remove sparse terms, and turn it into a data frame. 
# We selected one particular threshold to remove sparse terms, but remember that you can try different numbers!

dtm = DocumentTermMatrix(CorpusHeadline)

sparse = removeSparseTerms(dtm, 0.981)

sparse

HeadlineWords = as.data.frame(as.matrix(sparse))

# Let's make sure our variable names are okay for R:

colnames(HeadlineWords) = make.names(colnames(HeadlineWords))

str(HeadlineWords)

# Now we need to split the observations back into the training set and testing set.
# To do this, we can use the head and tail functions in R. 
# The head function takes the first "n" rows of HeadlineWords (the first argument to the head function), where "n" is specified by the second argument to the head function. 
# So here we are taking the first nrow(NewsTrain) observations from HeadlineWords, and putting them in a new data frame called "HeadlineWordsTrain"

HeadlineWordsTrain = head(HeadlineWords, nrow(NewsTrain))

# The tail function takes the last "n" rows of HeadlineWords (the first argument to the tail function), where "n" is specified by the second argument to the tail function. 
# So here we are taking the last nrow(NewsTest) observations from HeadlineWords, and putting them in a new data frame called "HeadlineWordsTest"

HeadlineWordsTest = tail(HeadlineWords, nrow(NewsTest))

# Note that this split of HeadlineWords works to properly put the observations back into the training and testing sets, because of how we combined them together when we first made our corpus.

# Before building models, we want to add back the original variables from our datasets. We'll add back the dependent variable to the training set, and the WordCount variable to both datasets. You might want to add back more variables to use in your model - we'll leave this up to you!

HeadlineWordsTrain$Popular = as.factor(NewsTrain$Popular)
HeadlineWordsTrain$NewsDesk=as.factor(NewsTrain$NewsDesk)
HeadlineWordsTrain$SectionName=as.factor(NewsTrain$SectionName)
HeadlineWordsTrain$SubsectionName=as.factor(NewsTrain$SubsectionName)
HeadlineWordsTrain$WordCount = NewsTrain$WordCount
HeadlineWordsTrain$Weekday = as.factor(NewsTrain$Weekday)
HeadlineWordsTrain$DayofMonth=as.factor(NewsTrain$DayofMonth)
HeadlineWordsTrain$Time=as.factor(NewsTrain$Time)


HeadlineWordsTest$WordCount = NewsTest$WordCount
HeadlineWordsTest$NewsDesk=as.factor(NewsTest$NewsDesk)
HeadlineWordsTest$SectionName=as.factor(NewsTest$SectionName)
HeadlineWordsTest$SubsectionName=as.factor(NewsTest$SubsectionName)
HeadlineWordsTest$Weekday = as.factor(NewsTest$Weekday)
HeadlineWordsTest$DayofMonth=as.factor(NewsTest$DayofMonth)
HeadlineWordsTest$Time=as.factor(NewsTest$Time)

library(sentiment)

sentiment(NewsTrain$Abstract)[,2]
HeadlineWordsTrain$Sentiment=sentiment(NewsTrain$Abstract)[,2]
HeadlineWordsTest$Sentiment=sentiment(NewsTest$Abstract)[,2]

HeadlineWordsTrain$Sentiment=as.factor(HeadlineWordsTrain$Sentiment)
HeadlineWordsTest$Sentiment=as.factor(HeadlineWordsTest$Sentiment)


library(caret)
library(e1071)
set.seed (32423)

gbmgrid=expand.grid(interaction.depth=c(4,5,7,8,9)*2,n.trees=(1:30)*50,shrinkage=0.001)
fitControl = trainControl (method = "cv", number = 10,classProbs = TRUE)
boostFit = train (Popular ~ ., method = "gbm", data = HeadlineWordsTrain, trControl = fitControl,tuneGrid=gbmgrid)


predgbm=predict(boostFit,newdata=HeadlineWordsTest,type="prob")[,2]

predgbm

MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = predgbm)

write.csv(MySubmission, "SubmissionHeadlineGBM_53.csv", row.names=FALSE)

print(boostFit)

plot(boostFit)

