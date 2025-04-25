// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title PremisesTemplate
 * @dev Represents the premises details from Section 2 of the Short-Term Rental Agreement.
 */
contract PremisesTemplate {
    string public mailingAddress;
    string public residenceType; // e.g., "Apartment", "House", "Condo", or specific type
    uint256 public numberOfBedrooms;
    uint256 public numberOfBathrooms;
    string public otherDescription; // Description of any other relevant features

    // --- Premises Details ---
    // mailingAddress: The full address of the property being leased.
    // residenceType: The type of dwelling.
    // numberOfBedrooms: Count of bedrooms.
    // numberOfBathrooms: Count of bathrooms.
    // otherDescription: Any additional description provided.

    // Note: This template primarily defines the data structure for the property description.
}