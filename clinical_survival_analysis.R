library(survival)
data <- read.csv('survival_data.csv')
fit <- survfit(Surv(time, status) ~ treatment, data=data)
summary(fit)
