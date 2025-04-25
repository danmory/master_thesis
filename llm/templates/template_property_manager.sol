// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title PropertyManagerTemplate
 * @dev Represents the property manager details from Section 11 of the Short-Term Rental Agreement.
 */
contract PropertyManagerTemplate {
    bool public hasPropertyManager;

    // Landlord contact info (if no manager)
    string public landlordPhoneNumber;
    string public landlordEmail;

    // Property Manager contact info (if manager exists)
    string public propertyManagerName;
    string public propertyManagerPhoneNumber;
    string public propertyManagerEmail;

    // --- Contact Details ---
    // hasPropertyManager: True if a property manager is designated, false otherwise.
    // landlordPhoneNumber: Landlord's phone (if no manager).
    // landlordEmail: Landlord's email (if no manager).
    // propertyManagerName: Manager's name (if manager exists).
    // propertyManagerPhoneNumber: Manager's phone (if manager exists).
    // propertyManagerEmail: Manager's email (if manager exists).

    // Note: This template primarily defines the data structure for contact information.
}