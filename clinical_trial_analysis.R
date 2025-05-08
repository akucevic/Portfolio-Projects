# Clinical Trial Analysis in R for Hormonal Contraception Effects

library(dplyr)
library(ggplot2)
library(tidyr)
library(lme4)
library(car)

# Load data
trial_data <- read.csv('clinical_trial.csv')

# Preprocess data
trial_data$Time <- factor(trial_data$Time, levels = c("Baseline", "3Months", "6Months"))
trial_data$Age_Group <- cut(trial_data$Age, breaks = c(18, 30, 45, 60), labels = c("18-30", "31-45", "46-60"))

# Summary statistics
summary_stats <- trial_data %>%
  group_by(Time, Race, Age_Group) %>%
  summarise(mean_glucose = mean(Glucose, na.rm = TRUE))

# Linear Mixed Model
model <- lmer(Glucose ~ Time + Age + Race + (1 | Subject_ID), data = trial_data)
summary(model)

# ANOVA for fixed effects
Anova(model)

# Visualization
ggplot(trial_data, aes(x=Time, y=Glucose, group=Subject_ID, color=Race)) +
  geom_line(alpha=0.3) +
  stat_summary(fun=mean, geom="line", aes(group=1), color="black", size=1.2) +
  theme_minimal()
