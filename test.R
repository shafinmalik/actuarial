library(ggplot2)
persinj <- read.csv("persinj.csv")
fig <- ggplot(data=persinj50, mapping=aes(x=op_time,y=amt)) + geom_point()
fig.show()