import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os.path
import scipy.stats as stats
from textblob import TextBlob
from nrclex  import NRCLex

vVaderAnalyser = SentimentIntensityAnalyzer()
vPreAnalysisData = pd.DataFrame()
vAllRatingsSentiments = pd.DataFrame()

def SentimentAnalyzerScores(pDialogue):
    vScore = vVaderAnalyser.polarity_scores(pDialogue)
    return dict(vScore)


def EmotionAnalyzerScores(pDialogue):
    vEmotions = NRCLex(pDialogue)
    return dict(vEmotions.affect_frequencies)

def CalcReviewMetrics(pMovieName, pTheaterReleaseDate, pVideoReleaseDate):

    dfReviews = pd.read_excel(f'Consolidated-Reviews-{pMovieName}.xlsx')

    dfReviews = dfReviews[['date', 'title', 'rating', 'body_x']]

    dfReviews = dfReviews.drop_duplicates()

    dfReviews['date'] = pd.to_datetime(dfReviews['date'], format="%m/%d/%Y")

    # Rating
    vRatingPreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['rating']]

    vAverageRatingPreVideo = vRatingPreVideo['rating'].mean()
    vTotalRatingPreVideo = len(vRatingPreVideo[['rating']])
    vSTDRatingPreVideo = vRatingPreVideo['rating'].std()

    vRatingPostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['rating']]

    vAverageRatingPostVideo = vRatingPostVideo['rating'].mean()
    vCountRatingPostVideo = len(vRatingPostVideo[['rating']])
    vSTDRatingPostVideo = vRatingPostVideo['rating'].std()

    # VADER sentiment
    dfReviews['body_x'] = dfReviews['body_x'].astype(str)
    dfReviews['Sentiment'] = dfReviews.apply(lambda vRow: SentimentAnalyzerScores(vRow['body_x']), axis=1)

    DFSentiment = pd.DataFrame(list(dfReviews['Sentiment']))
    DFSentiment.columns = ['Sentiment Negative', 'Sentiment Neutral', 'Sentiment Positive', 'Sentiment Compound']

    dfReviews.reset_index(inplace=True)
    dfReviews = pd.concat([dfReviews, DFSentiment], axis=1)
    dfReviews.drop(['Sentiment'], axis=1, inplace=True)

    dfReviews['Average Sentiment'] = dfReviews['Sentiment Compound']

    vSentimentPreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Average Sentiment']]

    vAverageSentimentPreVideo = vSentimentPreVideo['Average Sentiment'].mean()

    vTotalSentimentPreVideo = len(vSentimentPreVideo[['Average Sentiment']])
    vSTDSentimentPreVideo = vSentimentPreVideo['Average Sentiment'].std()

    vSentimentPostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Average Sentiment']]

    vAverageSentimentPostVideo = vSentimentPostVideo['Average Sentiment'].mean()
    vTotalSentimentPostVideo = len(vSentimentPostVideo[['Average Sentiment']])
    vSTDSentimentPostVideo = vSentimentPostVideo['Average Sentiment'].std()

    pTStatRating, pPValueRating = stats.ttest_ind(a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['rating']],
                                                  b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['rating']], equal_var=True)

    pTStatSentiment, pPValueSentiment = stats.ttest_ind(a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Average Sentiment']],
                                                        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Average Sentiment']], equal_var=True)

    # Correlation
    vRatingSentimentCorr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Average Sentiment'].to_list())

    # TextBlob sentiment
    dfReviews['Sentiment2'] = dfReviews.apply(lambda vRow: TextBlob(vRow['body_x']).sentiment.polarity, axis=1)

    vSentiment2PreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Sentiment2']]
    vAverageSentiment2PreVideo = vSentiment2PreVideo['Sentiment2'].mean()
    vSTDSentiment2PreVideo = vSentiment2PreVideo['Sentiment2'].std()

    vSentiment2PostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Sentiment2']]
    vAverageSentiment2PostVideo = vSentiment2PostVideo['Sentiment2'].mean()
    vSTDSentiment2PostVideo = vSentiment2PostVideo['Sentiment2'].std()

    pTStatSentiment2, pPValueSentiment2 = stats.ttest_ind(a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Sentiment2']],
                                                        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Sentiment2']], equal_var=True)

    vAllRatingsMovie = dfReviews['rating'].to_list()
    vAllSentimentsMovie = dfReviews['Average Sentiment'].to_list()
    vAllSentiments2Movie = dfReviews['Sentiment2'].to_list()

    vRatingSentiment2Corr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Sentiment2'].to_list())

    # Emotions
    dfReviews['Emotions'] = dfReviews.apply(lambda vRow: EmotionAnalyzerScores(vRow['body_x']), axis=1)

    DFEmotions = pd.DataFrame(list(dfReviews['Emotions']))
    DFEmotions.columns = ['Fear', 'Anger', 'Anticip', 'Trust', 'Surprise', 'Positive', 'Negative', 'Sadness', 'Disgust', 'Joy', 'Anticipation']

    dfReviews.reset_index(inplace=True)
    dfReviews = pd.concat([dfReviews, DFEmotions], axis=1)
    dfReviews.drop(['Emotions'], axis=1, inplace=True)

    # Anger
    vAllAngerMovie = dfReviews['Anger'].to_list()

    vAngerPreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Anger']]

    vAverageAngerPreVideo = vAngerPreVideo['Anger'].mean()
    vSTDAngerPreVideo = vAngerPreVideo['Anger'].std()

    vAngerPostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Anger']]

    vAverageAngerPostVideo = vAngerPostVideo['Anger'].mean()
    vSTDAngerPostVideo = vAngerPostVideo['Anger'].std()

    vRatingAngerCorr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Anger'].to_list())

    pTStatAnger, pPValueAnger = stats.ttest_ind(
        a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Anger']],
        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Anger']], equal_var=True)

    # Surprise
    vAllSurpriseMovie = dfReviews['Surprise'].to_list()

    vSurprisePreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Surprise']]

    vAverageSurprisePreVideo = vSurprisePreVideo['Surprise'].mean()
    vSTDSurprisePreVideo = vSurprisePreVideo['Surprise'].std()

    vSurprisePostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Surprise']]

    vAverageSurprisePostVideo = vSurprisePostVideo['Surprise'].mean()
    vSTDSurprisePostVideo = vSurprisePostVideo['Surprise'].std()

    vRatingSurpriseCorr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Surprise'].to_list())

    pTStatSurprise, pPValueSurprise = stats.ttest_ind(
        a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Surprise']],
        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Surprise']], equal_var=True)

    # Joy
    vAllJoyMovie = dfReviews['Joy'].to_list()

    vJoyPreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Joy']]

    vAverageJoyPreVideo = vJoyPreVideo['Joy'].mean()
    vSTDJoyPreVideo = vJoyPreVideo['Joy'].std()

    vJoyPostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Joy']]

    vAverageJoyPostVideo = vJoyPostVideo['Joy'].mean()
    vSTDJoyPostVideo = vJoyPostVideo['Joy'].std()

    vRatingJoyCorr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Joy'].to_list())

    pTStatJoy, pPValueJoy = stats.ttest_ind(
        a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Joy']],
        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Joy']], equal_var=True)

    # Fear
    vAllFearMovie = dfReviews['Fear'].to_list()

    vFearPreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Fear']]

    vAverageFearPreVideo = vFearPreVideo['Fear'].mean()
    vSTDFearPreVideo = vFearPreVideo['Fear'].std()

    vFearPostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Fear']]

    vAverageFearPostVideo = vFearPostVideo['Fear'].mean()
    vSTDFearPostVideo = vFearPostVideo['Fear'].std()

    vRatingFearCorr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Fear'].to_list())

    pTStatFear, pPValueFear = stats.ttest_ind(
        a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Fear']],
        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Fear']], equal_var=True)

    # Anticipation
    vAllAnticipationMovie = dfReviews['Anticipation'].to_list()

    vAnticipationPreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Anticipation']]

    vAverageAnticipationPreVideo = vAnticipationPreVideo['Anticipation'].mean()
    vSTDAnticipationPreVideo = vAnticipationPreVideo['Anticipation'].std()

    vAnticipationPostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Anticipation']]

    vAverageAnticipationPostVideo = vAnticipationPostVideo['Anticipation'].mean()
    vSTDAnticipationPostVideo = vAnticipationPostVideo['Anticipation'].std()

    vRatingAnticipationCorr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Anticipation'].to_list())

    pTStatAnticipation, pPValueAnticipation = stats.ttest_ind(
        a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Anticipation']],
        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Anticipation']], equal_var=True)

    # Trust
    vAllTrustMovie = dfReviews['Trust'].to_list()

    vTrustPreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Trust']]

    vAverageTrustPreVideo = vTrustPreVideo['Trust'].mean()
    vSTDTrustPreVideo = vTrustPreVideo['Trust'].std()

    vTrustPostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Trust']]

    vAverageTrustPostVideo = vTrustPostVideo['Trust'].mean()
    vSTDTrustPostVideo = vTrustPostVideo['Trust'].std()

    vRatingTrustCorr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Trust'].to_list())

    pTStatTrust, pPValueTrust = stats.ttest_ind(
        a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Trust']],
        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Trust']], equal_var=True)

    # Sadness
    vAllSadnessMovie = dfReviews['Sadness'].to_list()

    vSadnessPreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Sadness']]

    vAverageSadnessPreVideo = vSadnessPreVideo['Sadness'].mean()
    vSTDSadnessPreVideo = vSadnessPreVideo['Sadness'].std()

    vSadnessPostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Sadness']]

    vAverageSadnessPostVideo = vSadnessPostVideo['Sadness'].mean()
    vSTDSadnessPostVideo = vSadnessPostVideo['Sadness'].std()

    vRatingSadnessCorr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Sadness'].to_list())

    pTStatSadness, pPValueSadness = stats.ttest_ind(
        a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Sadness']],
        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Sadness']], equal_var=True)

    # Disgust
    vAllDisgustMovie = dfReviews['Disgust'].to_list()

    vDisgustPreVideo = dfReviews.loc[
        dfReviews['date'] <= pVideoReleaseDate, ['Disgust']]

    vAverageDisgustPreVideo = vDisgustPreVideo['Disgust'].mean()
    vSTDDisgustPreVideo = vDisgustPreVideo['Disgust'].std()

    vDisgustPostVideo = dfReviews.loc[
        dfReviews['date'] > pVideoReleaseDate, ['Disgust']]

    vAverageDisgustPostVideo = vDisgustPostVideo['Disgust'].mean()
    vSTDDisgustPostVideo = vDisgustPostVideo['Disgust'].std()

    vRatingDisgustCorr = np.corrcoef(dfReviews['rating'].to_list(), dfReviews['Disgust'].to_list())

    pTStatDisgust, pPValueDisgust = stats.ttest_ind(
        a=dfReviews.loc[dfReviews['date'] <= pVideoReleaseDate, ['Disgust']],
        b=dfReviews.loc[dfReviews['date'] > pVideoReleaseDate, ['Disgust']], equal_var=True)

    # Compile rows
    dfRow = pd.DataFrame({'Movie': pMovieName,
                          'Theatrical Release': pTheaterReleaseDate,
                          'Video Release': pVideoReleaseDate,
                          'Avg Rating Pre Video': vAverageRatingPreVideo,
                          'Total Rating Pre Video': vTotalRatingPreVideo,
                          'Variance Rating Pre Video': vSTDRatingPreVideo,
                          'Avg Rating Post Video': vAverageRatingPostVideo,
                          'Total Rating Post Video': vCountRatingPostVideo,
                          'Variance Rating Post Video': vSTDRatingPostVideo,
                          'Avg Sentiment Pre Video': vAverageSentimentPreVideo,
                          'Total Sentiment Pre Video': vTotalSentimentPreVideo,
                          'Variance Sentiment Pre Video': vSTDSentimentPreVideo,
                          'Avg Sentiment Post Video': vAverageSentimentPostVideo,
                          'Total Sentiment Post Video': vTotalSentimentPostVideo,
                          'Variance Sentiment Post Video': vSTDSentimentPostVideo,
                          'Avg Sentiment2 Pre Video': vAverageSentiment2PreVideo,
                          'Variance Sentiment2 Pre Video': vSTDSentiment2PreVideo,
                          'Avg Sentiment2 Post Video': vAverageSentiment2PostVideo,
                          'Variance Sentiment2 Post Video': vSTDSentiment2PostVideo,
                          'Avg Anger Pre Video': vAverageAngerPreVideo,
                          'Variance Anger Pre Video': vSTDAngerPreVideo,
                          'Avg Anger Post Video': vAverageAngerPostVideo,
                          'Variance Anger Post Video': vSTDAngerPostVideo,
                          'Avg Surprise Pre Video': vAverageSurprisePreVideo,
                          'Variance Surprise Pre Video': vSTDSurprisePreVideo,
                          'Avg Surprise Post Video': vAverageSurprisePostVideo,
                          'Variance Surprise Post Video': vSTDSurprisePostVideo,
                          'Avg Joy Pre Video': vAverageJoyPreVideo,
                          'Variance Joy Pre Video': vSTDJoyPreVideo,
                          'Avg Joy Post Video': vAverageJoyPostVideo,
                          'Variance Joy Post Video': vSTDJoyPostVideo,
                          'Avg Fear Pre Video': vAverageFearPreVideo,
                          'Variance Fear Pre Video': vSTDFearPreVideo,
                          'Avg Fear Post Video': vAverageFearPostVideo,
                          'Variance Fear Post Video': vSTDFearPostVideo,
                          'Avg Anticipation Pre Video': vAverageAnticipationPreVideo,
                          'Variance Anticipation Pre Video': vSTDAnticipationPreVideo,
                          'Avg Anticipation Post Video': vAverageAnticipationPostVideo,
                          'Variance Anticipation Post Video': vSTDAnticipationPostVideo,
                          'Avg Trust Pre Video': vAverageTrustPreVideo,
                          'Variance Trust Pre Video': vSTDTrustPreVideo,
                          'Avg Trust Post Video': vAverageTrustPostVideo,
                          'Variance Trust Post Video': vSTDTrustPostVideo,
                          'Avg Sadness Pre Video': vAverageSadnessPreVideo,
                          'Variance Sadness Pre Video': vSTDSadnessPreVideo,
                          'Avg Sadness Post Video': vAverageSadnessPostVideo,
                          'Variance Sadness Post Video': vSTDSadnessPostVideo,
                          'Avg Disgust Pre Video': vAverageDisgustPreVideo,
                          'Variance Disgust Pre Video': vSTDDisgustPreVideo,
                          'Avg Disgust Post Video': vAverageDisgustPostVideo,
                          'Variance Disgust Post Video': vSTDDisgustPostVideo,
                          'p-value Rating': pPValueRating.item(),
                          'p-value Sentiment': pPValueSentiment.item(),
                          'p-value Sentiment2': pPValueSentiment2.item(),
                          'p-value Anger': pPValueAnger.item(),
                          'p-value Surprise': pPValueSurprise.item(),
                          'p-value Joy': pPValueJoy.item(),
                          'p-value Fear': pPValueFear.item(),
                          'p-value Anticipation': pPValueAnticipation.item(),
                          'p-value Trust': pPValueTrust.item(),
                          'p-value Sadness': pPValueSadness.item(),
                          'p-value Disgust': pPValueDisgust.item(),
                          'Rating Sentiment Corr': vRatingSentimentCorr[0, 1],
                          'Rating Sentiment2 Corr': vRatingSentiment2Corr[0, 1],
                          'Rating Anger Corr': vRatingAngerCorr[0, 1],
                          'Rating Surprise Corr': vRatingSurpriseCorr[0, 1],
                          'Rating Joy Corr': vRatingJoyCorr[0, 1],
                          'Rating Fear Corr': vRatingFearCorr[0, 1],
                          'Rating Anticipation Corr': vRatingAnticipationCorr[0, 1],
                          'Rating Trust Corr': vRatingTrustCorr[0, 1],
                          'Rating Sadness Corr': vRatingSadnessCorr[0, 1],
                          'Rating Disgust Corr': vRatingDisgustCorr[0, 1]
                          }, index=[0])

    return [dfRow,
            vAllRatingsMovie,
            vAllSentimentsMovie,
            vAllSentiments2Movie,
            vAllAngerMovie,
            vAllSurpriseMovie,
            vAllJoyMovie,
            vAllFearMovie,
            vAllAnticipationMovie,
            vAllTrustMovie,
            vAllSadnessMovie,
            vAllDisgustMovie]

# Main code
vMovieKeys = pd.read_excel('Amazon Reviews Full List.xlsx')
vMovieDates = pd.read_excel('MOVIE_JAN18_TO_FEB20_TNUM.xlsx')

vMergedData = pd.merge(vMovieKeys, vMovieDates[['BOMJ_RELEASE_DAYBO', 'TNUM_Video_ReleaseDate', 'MOJO_title']], left_on='Movie Name', right_on='MOJO_title', how='left')

vMergedData['BOMJ_RELEASE_DAYBO'] = pd.to_datetime(vMergedData['BOMJ_RELEASE_DAYBO'], format="%m/%d/%Y")
vMergedData['TNUM_Video_ReleaseDate'] = pd.to_datetime(vMergedData['TNUM_Video_ReleaseDate'], format="%m/%d/%Y")
#vMergedData['TNUM_Video_ReleaseEndDate'] = vMergedData['TNUM_Video_ReleaseDate'] + datetime.timedelta(days=364)

vMergedData = vMergedData.drop(['MOJO_title'], axis=1)

for vIndex, vRow in vMergedData.iterrows():

    vMovieName = vRow['Movie Name']
    vTheaterReleaseDate = vRow['BOMJ_RELEASE_DAYBO']
    vVideoReleaseDate = vRow['TNUM_Video_ReleaseDate']
    #vVideoReleaseEndDate = vRow['TNUM_Video_ReleaseEndDate']

    if (os.path.exists(f'Consolidated-Reviews-{vMovieName}.xlsx')):
        print(f'Appending {vMovieName}')
        #vPreAnalysisData = vPreAnalysisData.append(CalcReviewMetrics(vMovieName, vTheaterReleaseDate, vVideoReleaseDate, vVideoReleaseEndDate), ignore_index=True)

        vCalcReviewMetrics = CalcReviewMetrics(vMovieName, vTheaterReleaseDate, vVideoReleaseDate)

        vPreAnalysisData = vPreAnalysisData.append(
            vCalcReviewMetrics[0],
            ignore_index=True)

        dfTempDataFrame = pd.DataFrame(
            {
                'Rating': vCalcReviewMetrics[1],
                'Sentiment': vCalcReviewMetrics[2],
                'Sentiment2': vCalcReviewMetrics[3],
                'Anger': vCalcReviewMetrics[4],
                'Surprise': vCalcReviewMetrics[5],
                'Joy': vCalcReviewMetrics[6],
                'Fear': vCalcReviewMetrics[7],
                'Anticipation': vCalcReviewMetrics[8],
                'Trust': vCalcReviewMetrics[9],
                'Sadness': vCalcReviewMetrics[10],
                'Disgust': vCalcReviewMetrics[11]
            }
        )

        vAllRatingsSentiments = vAllRatingsSentiments.append(dfTempDataFrame)

"""
        vAllRatings = vAllRatings.append(
            vCalcReviewMetrics[1]
        )

        vAllSentiments = vAllSentiments.append(
            vCalcReviewMetrics[2]
        )

        vAllSentiments2 = vAllSentiments.append(
            vCalcReviewMetrics[3]
        )
"""
vAllRatingsSentiments.to_excel('vAllRatingsSentiments.xlsx')
vPreAnalysisData.to_excel('PreAnalysis.xlsx')