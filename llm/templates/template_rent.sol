// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title RentTemplate
 * @dev Represents the rent details from Section 5 of the Short-Term Rental Agreement.
 */
contract RentTemplate {
    enum RentPaymentType { FixedAmount, MonthlyAmount }

    RentPaymentType public rentPaymentType;
    uint256 public totalRentAmount; // For FixedAmount
    uint256 public monthlyRentAmount; // For MonthlyAmount
    uint256 public rentDueDateDay; // Day of the month rent is due (for MonthlyAmount)
    string public paymentInstructions; // For MonthlyAmount

    // --- Fixed Amount Details ---
    // totalRentAmount: The total rent for the entire lease term.

    // --- Monthly Amount Details ---
    // monthlyRentAmount: The amount due each month.
    // rentDueDateDay: The day of the month the rent payment is due.
    // paymentInstructions: How the tenant should make the monthly payments.

    // Note: Actual implementation would involve functions for payment processing, tracking due dates, etc.
    // This template primarily defines the data structure.
}