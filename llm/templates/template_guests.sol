contract GuestsTemplate {
    bool public guestsAllowed = true;
    uint256 public maxGuestCount = 5; // maximum 5 guests allowed
    uint256 public maxGuestStayHours = 2; // guests can stay 2 hours
    string public otherGuestRules = "any other requirements";
}