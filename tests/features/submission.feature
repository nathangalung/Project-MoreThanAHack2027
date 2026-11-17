Feature: Submission File
  One label per video frame.

  Scenario: Every Frame Gets a Label
    Given a match with two graded halves
    And trained and adapted networks
    When the prediction runs
    Then the file has one row per frame
    And frames outside the halves are X
    And graded frames are H, A, or D

  Scenario: Play Is Dead Before Each Kickoff
    Given a match with two graded halves
    And trained and adapted networks
    When the prediction runs
    Then every graded frame up to a kickoff is D

  Scenario: The File Does Not Need Adapted Networks
    Given a match with two graded halves
    And trained and adapted networks
    When the prediction runs without the adapted networks
    Then the file has one row per frame
    And no adapted weights were needed

  Scenario: Adapted Networks Only Split the Teams
    Given frame probabilities from the trained networks
    And frame probabilities from the adapted networks
    When the team split is taken from the adapted networks
    Then the dead ball probability is unchanged
    And the live probability follows the adapted team split

  Scenario: Networks Adapt on Confident Pseudo-Labels
    Given a match with two graded halves
    And trained and adapted networks
    When the networks adapt to the match
    Then adapted weights are saved for every network
    And frames before a kickoff are left out of the pseudo-labels
    And frames under the confidence threshold are left out
