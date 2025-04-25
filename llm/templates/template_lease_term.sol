// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title LeaseTermTemplate
 * @dev Represents the lease term details from Section 3 of the Short-Term Rental Agreement.
 */
contract LeaseTermTemplate {
    enum LeaseType { FixedTerm, MonthToMonth }

    LeaseType public leaseType;
    uint256 public startDate; // Unix timestamp
    string public checkInTime; // e.g., "15:00"
    uint256 public endDate; // Unix timestamp (for FixedTerm)
    string public checkOutTime; // e.g., "11:00" (for FixedTerm)
    uint256 public noticePeriodDays; // Days (for MonthToMonth)

    // --- Fixed Term Details ---
    // startDate
    // checkInTime
    // endDate
    // checkOutTime

    // --- Month-to-Month Details ---
    // startDate
    // noticePeriodDays

    // Note: Actual implementation would involve functions to set and manage these details.
    // This template primarily defines the data structure.
}