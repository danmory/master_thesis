// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title GuestsTemplate
 * @dev Represents the guest policy details from Section 13 of the Short-Term Rental Agreement.
 */
contract GuestsTemplate {
    bool public guestsAllowed;
    uint256 public maxNumberOfGuests;
    uint256 public maxStayHours; // Maximum duration of a guest's stay in hours.
    string public otherGuestRulesDescription;

    // --- Guest Policy Details ---
    // guestsAllowed: True if guests are permitted, false otherwise.
    // maxNumberOfGuests: The maximum number of guests allowed at any one time.
    // maxStayHours: The maximum length of time a guest can stay.
    // otherGuestRulesDescription: Any additional rules specified for guests.

    // Note: This template primarily defines the data structure.
}