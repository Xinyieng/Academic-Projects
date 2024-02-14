rm(list=ls())

#1. Identifying the highest occuring noun
install.packages("udpipe")
install.packages("textrank")
library(udpipe)
library(textrank)
## First step: Take the Spanish udpipe model and annotate the text. Note: this takes about 3 minute
growth=read.csv("management_dataset_all.csv", stringsAsFactors = FALSE)
ud_model <- udpipe_download_model(language = "english")
str(ud_model)
ud_model <- udpipe_load_model(ud_model$file_model)
x <- udpipe_annotate(ud_model, x = growth$ï..Sentences)
x <- as.data.frame(x)

stats <- subset(x, upos %in% "NOUN")
stats <- txt_freq(x = stats$lemma)
library(lattice)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 10), col = "cadetblue", main = "Most occurring nouns", xlab = "Freq")

#4.
stats <- keywords_rake(x = x, 
                       term = "token", group = c("doc_id", "paragraph_id", "sentence_id"),
                       relevant = x$upos %in% c("NOUN", "ADJ"),
                       ngram_max = 1)
head(subset(stats, freq > 3))

#5.
## Simple noun phrases (a adjective+noun, pre/postposition, optional determiner and another adjective+noun)
x$phrase_tag <- as_phrasemachine(x$upos, type = "upos")
stats <- keywords_phrases(x = x$phrase_tag, term = x$token, 
                          pattern = "(A|N)+N(P+D*(A|N)*N)*", 
                          is_regex = TRUE, ngram_max = 2, detailed = FALSE)
head(subset(stats, ngram > 1))
