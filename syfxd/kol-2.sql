


-- CREATE VIEW [Lata działalności]
-- AS
--     SELECT DISTINCT Year(O.OrderDate) AS Year
--     FROM Orders O

-- GO

-- SELECT * FROM [Lata działalności]


-- SELECT * FROM Customers C
-- WHERE C.CustomerID NOT IN (
--     SELECT O.CustomerID
--     FROM Orders O
--     WHERE YEAR(O.OrderDate) = 2024
--       AND O.CustomerID IS NOT NULL
-- )

-- Select CustomerID, count(rok)
-- from
-- (
-- SELECT DISTINCT C.CustomerID, Year(O.OrderDate) rok
-- from Customers C JOIN Orders O on O.CustomerID=C.CustomerID
-- WHERE YEAR(O.OrderDate) <= 1998
-- ) as sub
-- GROUP BY CustomerID having max(rok) != 1998 and count(rok) = 
-- (Select count(DISTINCT Year(Orders.OrderDate))
--     from Orders
--     WHERE Year(Orders.OrderDate) <=1998
-- ) -1

-- SELECT 
--     C.CustomerID, 
--     LD.[Year],
--     ISNULL(SUM(OD.UnitPrice * OD.Quantity * (1 - OD.Discount)), 0) AS TotalAmount
-- FROM Customers C 
-- CROSS JOIN [Lata działalności] LD
-- LEFT JOIN Orders O ON C.CustomerID = O.CustomerID 
--     AND YEAR(O.OrderDate) = LD.[Year]
-- LEFT JOIN [Order Details] OD ON O.OrderID = OD.OrderID
-- GROUP BY C.CustomerID, LD.[Year]
-- ORDER BY C.CustomerID, LD.[Year]


Select kraje.*, Customers.CustomerID from
(
SELECT Country, AVG(kwota) średnia, min(kwota) minimalna, max(kwota) maksymalna
FROM
(
SELECT C.CustomerID, C.country, SUM(OD.UnitPrice * OD.Quantity * (1 - OD.Discount)) kwota
FROM Customers C LEFT JOIN Orders O on C.CustomerID = O.CustomerID LEFT JOIN [Order Details] OD on OD.OrderID = O.OrderID
WHERE OD.UnitPrice is NOT NULL
GROUP BY C.CustomerID, C.Country
) sub
GROUP by country
) kraje
JOIN Customers on kraje.Country=Customers.Country