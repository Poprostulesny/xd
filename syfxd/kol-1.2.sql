-- SELECT CustomerID, sum(pomoc) from
-- (
-- Select C.CustomerID, P.CategoryID,
-- CASE WHEN P.CategoryID = 1 or P.CategoryID is NULL then 0 else 1 end as pomoc
-- from Customers C left join Orders O on C.CustomerID=O.CustomerID left join [Order Details] OD on OD.OrderID=O.OrderID left join Products P on P.ProductID = OD.ProductID
-- ) as sub
-- GROUP by CustomerID

SELECT C.CustomerID, O.OrderDate, O.OrderID
FROM Customers C 
JOIN Orders O ON C.CustomerID = O.CustomerID
WHERE C.CustomerID IN
(
    SELECT TOP 3 O2.CustomerID 
    FROM Orders O2 
    JOIN [Order Details] OD2 ON O2.OrderID = OD2.OrderID
    GROUP BY O2.CustomerID
    ORDER BY SUM(ROUND(UnitPrice*Quantity*(1-CONVERT(MONEY,Discount)),2)) DESC
)
AND 
(
    SELECT COUNT(*) 
    FROM Orders O3 
    WHERE ((O3.OrderDate > O.OrderDate) 
        OR (O3.OrderDate = O.OrderDate AND O3.OrderID > O.OrderID))
        AND O3.CustomerID = C.CustomerID
) < 2
ORDER BY C.CustomerID, O.OrderDate DESC, O.OrderID DESC