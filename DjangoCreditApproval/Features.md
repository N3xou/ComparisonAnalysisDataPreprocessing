Feature changelog:
Income scaling
  Results:
    LRC - Different results, shifted prediction from 1's to 0's
    DTC - Better results
    RFC - Worse results 
Income grouping (4) (one-hot encoding)
  Results:
    LRC,DTC,RFC - Minor improvements
    Between 3,4,5 diff results not necessairly improvement  
Income grouping (4) (ORDINAL)
  Results:
    LRC - Minor improvement from one-hot
    DTC - No improvement, different results
    RFC - No improvement, different results
Age grouping (3-5)
  Results:
    LRC,DTC,RFC - NO visible improvements, shift in data prediction
  Note:
    Grouping age shifts the predictions towards 0 
Employment days grouping (3-5)
  Results:
    Same as age grouping
    For groups of 10, improvement in RFC
Education changing one-hot to ordinal:
  Results:
    LRC - Visible improvment
    DTC - Shift in prediction to 0
    RFC - Shift in prediction, looks like improvement
Handling edge cases of children/family:
  Results:
    LRC - Minor improvment
    DTC - Shift in prediction to 0
    RFC - Minor improvement


