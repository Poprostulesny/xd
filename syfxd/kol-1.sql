-- SELECT * from Customers
-- SELECT * FROM [Order Details]
-- SELECT * FROM Orders
-- SELECT * from Products


-- SELECT DISTINCT C.*
-- FROM Customers C JOIN Orders O on C.CustomerID = O.CustomerID JOIN [Order Details] OD on O.OrderID = OD.OrderID JOIN Products P on P.ProductID = OD.ProductID
-- WHERE p.ProductID = 1 and C.Country = 'Poland'


-- SELECT C.CustomerID, COUNT(O.OrderID) AS OrderCount
-- FROM Customers C 
-- LEFT JOIN Orders O ON C.CustomerID = O.CustomerID 
--     AND YEAR(O.OrderDate) <= 1996
-- LEFT JOIN [Order Details] OD ON O.OrderID = OD.OrderID 
-- LEFT JOIN Products P ON P.ProductID = OD.ProductID 
--     AND P.ProductID = 1
-- GROUP BY C.CustomerID
-- ORDER BY OrderCount DESC

-- SELECT DISTINCT C.CustomerID, C.CompanyName
-- FROM Customers C
-- WHERE C.CustomerID IN (
--     -- Klienci którzy NIE zamówili napoju przed 1997
--     SELECT O.CustomerID
--     FROM Orders O
--     WHERE YEAR(O.OrderDate) < 1997
--       AND O.CustomerID NOT IN (
--           SELECT O2.CustomerID
--           FROM Orders O2
--           JOIN [Order Details] OD ON O2.OrderID = OD.OrderID
--           WHERE YEAR(O2.OrderDate) < 1997
--             AND OD.ProductID = 1
--       )
-- )
-- AND C.CustomerID IN (
--     -- Klienci którzy zamówili napój po 1997
--     SELECT O.CustomerID
--     FROM Orders O
--     JOIN [Order Details] OD ON O.OrderID = OD.OrderID
--     WHERE YEAR(O.OrderDate) >= 1997
--       AND OD.ProductID = 1
-- )

SELECT O.OrderID, O.OrderDate, P.ProductID, P.ProductName
FROM Orders O 
JOIN [Order Details] OD ON O.OrderID = OD.OrderID 
JOIN Products P ON P.ProductID = OD.ProductID
WHERE P.ProductID IN (
    SELECT TOP 3 P.ProductID
    FROM Products P
    ORDER BY P.UnitPrice DESC
)
AND (
    SELECT COUNT(O2.OrderDate)
    FROM Orders O2
    JOIN [Order Details] OD2 ON O2.OrderID = OD2.OrderID
    WHERE OD2.ProductID = P.ProductID
      AND O2.OrderDate >= O.OrderDate
) <= 2
ORDER BY P.ProductID, O.OrderDate DESC
